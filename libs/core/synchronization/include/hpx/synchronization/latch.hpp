//  Copyright (c) 2015-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/synchronization/latch.hpp

#pragma once

#include <hpx/assert.hpp>
#include <hpx/concurrency/cache_line_data.hpp>
#include <hpx/synchronization/detail/condition_variable.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/type_support/unused.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx {

    /// Latches are a thread coordination mechanism that allow one or more
    /// threads to block until an operation is completed. An individual latch
    /// is a singleuse object; once the operation has been completed, the latch
    /// cannot be reused.
    class latch
    {
    public:
        HPX_NON_COPYABLE(latch);

    protected:
        using mutex_type = hpx::spinlock;

    public:
        /// Initialize the latch
        ///
        /// Requires: count >= 0.
        /// Synchronization: None
        /// Postconditions: counter_ == count.
        ///
        explicit latch(std::ptrdiff_t count)
          : mtx_()
          , cond_()
          , counter_(count)
          , notified_(count == 0)
        {
        }

        /// Requires: No threads are blocked at the synchronization point.
        ///
        /// \note May be called even if some threads have not yet returned
        ///       from wait() or count_down_and_wait(), provided that counter_
        ///       is 0.
        /// \note The destructor might not return until all threads have exited
        ///       wait() or count_down_and_wait().
        /// \note It is the caller's responsibility to ensure that no other
        ///       thread enters wait() after one thread has called the
        ///       destructor. This may require additional coordination.
#if defined(HPX_DEBUG)
        ~latch()
        {
            HPX_ASSERT(counter_ == 0);
        }
#else
        ~latch() = default;
#endif

        /// Returns:        The maximum value of counter that the implementation
        ///                 supports.
        static constexpr std::ptrdiff_t(max)() noexcept
        {
            return (std::numeric_limits<std::ptrdiff_t>::max)();
        }

        /// Decrements counter_ by n. Does not block.
        ///
        /// Requires: counter_ >= n and n >= 0.
        ///
        /// Synchronization: Synchronizes with all calls that block on this
        /// latch and with all try_wait calls on this latch that return true .
        ///
        /// \throws Nothing.
        ///
        void count_down(std::ptrdiff_t update)
        {
            HPX_ASSERT(update >= 0);

            std::ptrdiff_t new_count = (counter_ -= update);
            HPX_ASSERT(new_count >= 0);

            // 26111: Caller failing to release lock 'this->mtx_.data_'
            // 26115: Failing to release lock 'this->mtx_.data_'
            // 26117: Releasing unheld lock 'this->mtx_.data_'
#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(disable : 26111 26115 26117)
#endif
            if (new_count == 0)
            {
                std::unique_lock l(mtx_.data_);
                notified_ = true;

                // Note: we use notify_one repeatedly instead of notify_all as we
                // know that our implementation of condition_variable::notify_one
                // relinquishes the lock before resuming the waiting thread
                // that avoids suspension of this thread when it tries to
                // re-lock the mutex while exiting from condition_variable::wait
                while (cond_.data_.notify_one(
                    HPX_MOVE(l), threads::thread_priority::boost))
                {
                    l = std::unique_lock(mtx_.data_);
                }
            }

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif
        }

        /// Returns:        With very low probability false. Otherwise
        ///                 counter == 0.
        bool try_wait() const noexcept
        {
            return counter_.load(std::memory_order_acquire) == 0;
        }

        /// If counter_ is 0, returns immediately. Otherwise, blocks the
        /// calling thread at the synchronization point until counter_
        /// reaches 0.
        ///
        /// \throws Nothing.
        ///
        void wait() const
        {
            // 26110: Caller failing to hold lock 'this->mtx_.data_'
            // 26117: Releasing unheld lock 'this->mtx_.data_'
#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(disable : 26110 26117)
#endif

            std::unique_lock l(mtx_.data_);
            if (counter_.load(std::memory_order_relaxed) > 0 || !notified_)
            {
                cond_.data_.wait(l, "hpx::latch::wait");

                HPX_ASSERT_LOCKED(
                    l, counter_.load(std::memory_order_relaxed) == 0);
                HPX_ASSERT_LOCKED(l, notified_);
            }

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif
        }

        /// Effects: Equivalent to:
        ///             count_down(update);
        ///             wait();
        void arrive_and_wait(std::ptrdiff_t update = 1)
        {
            HPX_ASSERT(update >= 0);

            std::unique_lock l(mtx_.data_);

            std::ptrdiff_t old_count =
                counter_.fetch_sub(update, std::memory_order_relaxed);
            HPX_ASSERT_LOCKED(l, old_count >= update);

            // 26110: Caller failing to hold lock 'this->mtx_.data_'
            // 26111: Caller failing to release lock 'this->mtx_.data_'
            // 26115: Failing to release lock 'this->mtx_.data_'
            // 26117: Releasing unheld lock 'this->mtx_.data_'
#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(disable : 26110 26111 26115 26117)
#endif

            if (old_count > update)
            {
                cond_.data_.wait(l, "hpx::latch::arrive_and_wait");

                HPX_ASSERT_LOCKED(
                    l, counter_.load(std::memory_order_relaxed) == 0);
                HPX_ASSERT_LOCKED(l, notified_);
            }
            else
            {
                notified_ = true;

                // Note: we use notify_one repeatedly instead of notify_all as we
                // know that our implementation of condition_variable::notify_one
                // relinquishes the lock before resuming the waiting thread
                // which avoids suspension of this thread when it tries to
                // re-lock the mutex while exiting from condition_variable::wait
                while (cond_.data_.notify_one(
                    HPX_MOVE(l), threads::thread_priority::boost))
                {
                    l = std::unique_lock(mtx_.data_);
                }
            }

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif
        }

    protected:
        mutable util::cache_line_data<mutex_type> mtx_;
        mutable util::cache_line_data<
            hpx::lcos::local::detail::condition_variable>
            cond_;
        std::atomic<std::ptrdiff_t> counter_;
        bool notified_;
    };
}    // namespace hpx

namespace hpx::lcos::local {

    /// \cond NOINTERNAL
    using cpp20_latch HPX_DEPRECATED_V(1, 8,
        "hpx::lcos::local::cpp20_latch is deprecated, use hpx::latch instead") =
        hpx::latch;
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// A latch maintains an internal counter_ that is initialized when the
    /// latch is created. Threads may block at a synchronization point waiting
    /// for counter_ to be decremented to 0. When counter_ reaches 0, all such
    /// blocked threads are released.
    ///
    /// Calls to countdown_and_wait() , count_down() , wait() , is_ready(),
    /// count_up() , and reset() behave as atomic operations.
    ///
    /// \note   A \a hpx::latch is not an LCO in the sense that it has no global
    /// id and it can't be triggered using the action (parcel) mechanism. Use
    /// hpx::distributed::latch instead if this is required. It is just a low
    /// level synchronization primitive allowing to synchronize a given number
    /// of \a threads.
    class latch : public hpx::latch
    {
    public:
        HPX_NON_COPYABLE(latch);

    public:
        /// Initialize the latch
        ///
        /// Requires: count >= 0.
        /// Synchronization: None
        /// Postconditions: counter_ == count.
        ///
        explicit latch(std::ptrdiff_t count)
          : hpx::latch(count)
        {
        }

        /// Requires: No threads are blocked at the synchronization point.
        ///
        /// \note May be called even if some threads have not yet returned
        ///       from wait() or count_down_and_wait(), provided that counter_
        ///       is 0.
        /// \note The destructor might not return until all threads have exited
        ///       wait() or count_down_and_wait().
        /// \note It is the caller's responsibility to ensure that no other
        ///       thread enters wait() after one thread has called the
        ///       destructor. This may require additional coordination.
        ~latch() = default;

        /// Decrements counter_ by 1 . Blocks at the synchronization point
        /// until counter_ reaches 0.
        ///
        /// Requires: counter_ > 0.
        ///
        /// Synchronization: Synchronizes with all calls that block on this
        /// latch and with all is_ready calls on this latch that return true.
        ///
        /// \throws Nothing.
        ///
        void count_down_and_wait()
        {
            hpx::latch::arrive_and_wait();
        }

        /// Returns: counter_ == 0. Does not block.
        ///
        /// \throws Nothing.
        ///
        bool is_ready() const noexcept
        {
            return hpx::latch::try_wait();
        }

        void abort_all()
        {
            // 26115: Failing to release lock 'this->mtx_.data_'
#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(disable : 26115)
#endif

            std::unique_lock l(mtx_.data_);
            cond_.data_.abort_all(HPX_MOVE(l));

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif
        }

        /// Increments counter_ by n. Does not block.
        ///
        /// Requires:  n >= 0.
        ///
        /// \throws Nothing.
        ///
        void count_up(std::ptrdiff_t n)
        {
            HPX_ASSERT(n >= 0);

            std::ptrdiff_t old_count =
                counter_.fetch_add(n, std::memory_order_acq_rel);

            HPX_ASSERT(old_count > 0);
            HPX_UNUSED(old_count);
        }

        /// Reset counter_ to n. Does not block.
        ///
        /// Requires:  n >= 0.
        ///
        /// \throws Nothing.
        void reset(std::ptrdiff_t n)
        {
            HPX_ASSERT(n >= 0);

            std::ptrdiff_t old_count =
                counter_.exchange(n, std::memory_order_acq_rel);

            HPX_ASSERT(old_count == 0);
            HPX_UNUSED(old_count);

            std::scoped_lock l(mtx_.data_);
            notified_ = false;
        }

        /// Effects: Equivalent to:
        ///             if (is_ready())
        ///                 reset(count);
        ///             count_up(n);
        /// Returns: true if the latch was reset
        bool reset_if_needed_and_count_up(
            std::ptrdiff_t n, std::ptrdiff_t count)
        {
            HPX_ASSERT(n >= 0);
            HPX_ASSERT(count >= 0);

            std::unique_lock l(mtx_.data_);

            if (notified_)
            {
                notified_ = false;

                std::ptrdiff_t old_count =
                    counter_.fetch_add(n + count, std::memory_order_relaxed);

                HPX_ASSERT(old_count == 0);
                HPX_UNUSED(old_count);

                return true;
            }

            std::ptrdiff_t old_count =
                counter_.fetch_add(n, std::memory_order_relaxed);

            HPX_ASSERT(old_count > 0);
            HPX_UNUSED(old_count);

            return false;
        }
    };
}    // namespace hpx::lcos::local

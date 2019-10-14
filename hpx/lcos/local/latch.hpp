//  Copyright (c) 2015-2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/lcos/local/latch.hpp

#if !defined(HPX_LCOS_LATCH_APR_18_2015_0925PM)
#define HPX_LCOS_LATCH_APR_18_2015_0925PM

#include <hpx/assertion.hpp>
#include <hpx/concurrency/cache_line_data.hpp>
#include <hpx/lcos/local/detail/condition_variable.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/type_support/unused.hpp>

#include <atomic>
#include <cstddef>
#include <mutex>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local
{
    /// Latches are a thread coordination mechanism that allow one or more
    /// threads to block until an operation is completed. An individual latch
    /// is a singleuse object; once the operation has been completed, the latch
    /// cannot be reused.
    ///
    /// A latch maintains an internal counter_ that is initialized when the
    /// latch is created. Threads may block at a synchronization point waiting
    /// for counter_ to be decremented to 0. When counter_ reaches 0, all such
    /// blocked threads are released.
    ///
    /// Calls to countdown_and_wait() , count_down() , wait() , is_ready(),
    /// count_up() , and reset() behave as atomic operations.
    ///
    /// \note   A \a local::latch is not a LCO in the sense that it has no
    ///         global id and it can't be triggered using the action (parcel)
    ///         mechanism. Use lcos::latch instead if this is required.
    ///         It is just a low level synchronization primitive allowing to
    ///         synchronize a given number of \a threads.
    class latch
    {
    public:
        HPX_NON_COPYABLE(latch);

    private:
        typedef lcos::local::spinlock mutex_type;

    public:
        /// Initialize the latch
        ///
        /// Requires: count >= 0.
        /// Synchronization: None
        /// Postconditions: counter_ == count.
        ///
        explicit latch(std::ptrdiff_t count)
          : mtx_(), cond_(), counter_(count), notified_(count == 0)
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
        ~latch ()
        {
            HPX_ASSERT(counter_ == 0);
        }

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
            std::unique_lock<mutex_type> l(mtx_.data_);
            HPX_ASSERT(counter_ > 0);
            if (--counter_ == 0)
            {
                notified_ = true;
                cond_.data_.notify_all(std::move(l));    // release the threads
            }
            else
                cond_.data_.wait(l, "hpx::local::latch::count_down_and_wait");
        }

        /// Decrements counter_ by n. Does not block.
        ///
        /// Requires: counter_ >= n and n >= 0.
        ///
        /// Synchronization: Synchronizes with all calls that block on this
        /// latch and with all is_ready calls on this latch that return true .
        ///
        /// \throws Nothing.
        ///
        void count_down(std::ptrdiff_t n)
        {
            HPX_ASSERT(n >= 0);

            std::ptrdiff_t new_count = (counter_ -= n);
            HPX_ASSERT(new_count >= 0);

            if (new_count == 0)
            {
                std::unique_lock<mutex_type> l(mtx_.data_);
                notified_ = true;
                cond_.data_.notify_all(std::move(l));    // release the threads
            }
        }

        /// Returns: counter_ == 0. Does not block.
        ///
        /// \throws Nothing.
        ///
        bool is_ready() const noexcept
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
            std::unique_lock<mutex_type> l(mtx_.data_);
            if (counter_.load(std::memory_order_relaxed) > 0 || !notified_)
                cond_.data_.wait(l, "hpx::local::latch::wait");
        }

        void abort_all()
        {
            std::unique_lock<mutex_type> l(mtx_.data_);
            cond_.data_.abort_all(std::move(l));
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

            std::unique_lock<mutex_type> l(mtx_.data_);
            HPX_ASSERT(notified_);
            notified_ = false;
        }

    protected:
        mutable util::cache_line_data<mutex_type> mtx_;
        mutable util::cache_line_data<local::detail::condition_variable> cond_;
        std::atomic<std::ptrdiff_t> counter_;
        bool notified_;
    };
}}}

#endif


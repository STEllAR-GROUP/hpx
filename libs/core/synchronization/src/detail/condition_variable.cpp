//  Copyright (c) 2007-2023 Hartmut Kaiser
//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/execution_base/agent_ref.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/synchronization/detail/condition_variable.hpp>
#include <hpx/synchronization/no_mutex.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/thread_support/assert_owns_lock.hpp>
#include <hpx/thread_support/unlock_guard.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/timing/steady_clock.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <exception>
#include <mutex>
#include <utility>

namespace hpx::lcos::local::detail {

    ///////////////////////////////////////////////////////////////////////////
    struct condition_variable::queue_entry
    {
        constexpr queue_entry(
            hpx::execution_base::agent_ref ctx, void* q) noexcept
          : ctx_(ctx)
          , q_(q)
        {
        }

        hpx::execution_base::agent_ref ctx_;
        void* q_;

        queue_entry* next = nullptr;
        queue_entry* prev = nullptr;
    };

    struct condition_variable::reset_queue_entry
    {
        explicit constexpr reset_queue_entry(
            condition_variable::queue_entry& e) noexcept
          : e_(e)
        {
        }

        reset_queue_entry(reset_queue_entry const&) = delete;
        reset_queue_entry(reset_queue_entry&&) = delete;
        reset_queue_entry& operator=(reset_queue_entry const&) = delete;
        reset_queue_entry& operator=(reset_queue_entry&&) = delete;

        ~reset_queue_entry()
        {
            if (e_.ctx_)
            {
                auto* q = static_cast<condition_variable::queue_type*>(e_.q_);
                q->erase(&e_);    // remove entry from queue
            }
        }

        condition_variable::queue_entry& e_;
    };

    ///////////////////////////////////////////////////////////////////////////
    condition_variable::~condition_variable()
    {
        if (!queue_.empty())
        {
            LERR_(fatal).format(
                "~condition_variable: queue is not empty, aborting threads");

            hpx::no_mutex no_mtx;
            std::unique_lock<hpx::no_mutex> lock(no_mtx);

            // Failing to release lock 'no_mtx' in function
#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(disable : 26115)
#endif
            abort_all<hpx::no_mutex>(HPX_MOVE(lock));
#if defined(HPX_MSVC)
#pragma warning(pop)
#endif
        }
    }

    bool condition_variable::empty(
        [[maybe_unused]] std::unique_lock<mutex_type>& lock) const noexcept
    {
        HPX_ASSERT_OWNS_LOCK(lock);
        return queue_.empty();
    }

    std::size_t condition_variable::size(
        [[maybe_unused]] std::unique_lock<mutex_type>& lock) const noexcept
    {
        HPX_ASSERT_OWNS_LOCK(lock);
        return queue_.size();
    }

    // Return false if no more threads are waiting (returns true if queue
    // is non-empty).
    bool condition_variable::notify_one(std::unique_lock<mutex_type>& lock,
        threads::thread_priority priority, bool unlock, error_code& ec)
    {
        // Caller failing to hold lock 'lock' before calling function
#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(disable : 26110)
#endif

        HPX_ASSERT_OWNS_LOCK(lock);

        if (!queue_.empty())
        {
            auto const ctx = queue_.front()->ctx_;

            // remove item from queue before error handling
            queue_.front()->ctx_.reset();
            queue_.pop_front();

            if (HPX_UNLIKELY(!ctx))
            {
                lock.unlock();

                HPX_THROWS_IF(ec, hpx::error::null_thread_id,
                    "condition_variable::notify_one",
                    "null thread id encountered");
                return false;
            }

            bool const not_empty = !queue_.empty();
            if (unlock)
                lock.unlock();

            ctx.resume();

            return not_empty;
        }

        if (&ec != &throws)
            ec = make_success_code();

        if (unlock)
            lock.unlock();

        return false;

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif
    }

    void condition_variable::notify_all(std::unique_lock<mutex_type> lock,
        threads::thread_priority /* priority */, error_code& ec)
    {
        // Caller failing to hold lock 'lock' before calling function
#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(disable : 26110)
#endif
        HPX_ASSERT_OWNS_LOCK(lock);

        // swap the list
        queue_type queue;
        queue.swap(queue_);

        if (!queue.empty())
        {
            // update reference to queue for all queue entries
            for (queue_entry* qe = queue_.front(); qe != nullptr; qe = qe->next)
            {
                qe->q_ = &queue;    //-V506
            }

            do
            {
                auto ctx = queue.front()->ctx_;

                // remove item from queue before error handling
                queue.front()->ctx_.reset();
                queue.pop_front();

                if (HPX_UNLIKELY(!ctx))
                {
                    prepend_entries(lock, queue);
                    lock.unlock();

                    HPX_THROWS_IF(ec, hpx::error::null_thread_id,
                        "condition_variable::notify_all",
                        "null thread id encountered");
                    return;
                }

                util::ignore_while_checking const il(&lock);
                HPX_UNUSED(il);

                ctx.resume();

            } while (!queue.empty());
        }

        if (&ec != &throws)
            ec = make_success_code();

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif
    }

    void condition_variable::abort_all(std::unique_lock<mutex_type> lock)
    {
        HPX_ASSERT_OWNS_LOCK(lock);

        abort_all<mutex_type>(HPX_MOVE(lock));
    }

    threads::thread_restart_state condition_variable::wait(
        std::unique_lock<mutex_type>& lock, char const* /* description */,
        error_code& /* ec */)
    {
        HPX_ASSERT_OWNS_LOCK(lock);

        // enqueue the request and block this thread
        auto const this_ctx = hpx::execution_base::this_thread::agent();
        queue_entry f(this_ctx, &queue_);
        queue_.push_back(f);

        reset_queue_entry r(f);
        {
            // suspend this thread
            unlock_guard<std::unique_lock<mutex_type>> ul(lock);
            this_ctx.suspend();
        }

        return f.ctx_ ? threads::thread_restart_state::timeout :
                        threads::thread_restart_state::signaled;
    }

    threads::thread_restart_state condition_variable::wait_until(
        std::unique_lock<mutex_type>& lock,
        hpx::chrono::steady_time_point const& abs_time,
        char const* /* description */, error_code& /* ec */)
    {
        HPX_ASSERT_OWNS_LOCK(lock);

        // enqueue the request and block this thread
        auto this_ctx = hpx::execution_base::this_thread::agent();
        queue_entry f(this_ctx, &queue_);
        queue_.push_back(f);

        reset_queue_entry r(f);
        {
            // suspend this thread
            unlock_guard<std::unique_lock<mutex_type>> ul(lock);
            this_ctx.sleep_until(abs_time.value());
        }

        return f.ctx_ ? threads::thread_restart_state::timeout :
                        threads::thread_restart_state::signaled;
    }

    template <typename Mutex>
    void condition_variable::abort_all(std::unique_lock<Mutex> lock)
    {
        // new threads might have been added while we were notifying
        while (!queue_.empty())
        {
            // swap the list
            queue_type queue;
            queue.swap(queue_);

            // update reference to queue for all queue entries
            for (queue_entry* qe = queue_.front(); qe != nullptr; qe = qe->next)
            {
                qe->q_ = &queue;    //-V506
            }

            while (!queue.empty())
            {
                auto ctx = queue.front()->ctx_;

                // remove item from queue before error handling
                queue.front()->ctx_.reset();
                queue.pop_front();

                if (HPX_UNLIKELY(!ctx))
                {
                    LERR_(fatal).format("condition_variable::abort_all: null "
                                        "thread id encountered");
                    continue;
                }

                LERR_(fatal).format(
                    "condition_variable::abort_all: pending thread: {}", ctx);

                // unlock while notifying thread as this can suspend
                unlock_guard<std::unique_lock<Mutex>> unlock(lock);

                // forcefully abort thread, do not throw
                ctx.abort();
            }
        }
    }

    // re-add the remaining items to the original queue
    void condition_variable::prepend_entries(
        [[maybe_unused]] std::unique_lock<mutex_type>& lock,
        queue_type& queue) noexcept
    {
        HPX_ASSERT_OWNS_LOCK(lock);
        queue.splice(queue_);
        queue_.swap(queue);
    }

    ///////////////////////////////////////////////////////////////////////////
    void intrusive_ptr_add_ref(condition_variable_data* p) noexcept
    {
        ++p->count_;
    }

    void intrusive_ptr_release(condition_variable_data* p) noexcept
    {
        if (0 == --p->count_)
        {
            delete p;
        }
    }
}    // namespace hpx::lcos::local::detail

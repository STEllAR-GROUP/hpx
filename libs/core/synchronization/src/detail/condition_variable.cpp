//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/synchronization/detail/condition_variable.hpp>
#include <hpx/synchronization/no_mutex.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/thread_support/unlock_guard.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <cstddef>
#include <exception>
#include <mutex>
#include <utility>

namespace hpx { namespace lcos { namespace local { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    condition_variable::condition_variable() {}

    condition_variable::~condition_variable()
    {
        if (!queue_.empty())
        {
            LERR_(fatal) << "~condition_variable: queue is not empty, "
                            "aborting threads";

            local::no_mutex no_mtx;
            std::unique_lock<local::no_mutex> lock(no_mtx);
            abort_all<local::no_mutex>(std::move(lock));
        }
    }

    bool condition_variable::empty(
        std::unique_lock<mutex_type> const& lock) const
    {
        HPX_ASSERT(lock.owns_lock());
        HPX_UNUSED(lock);

        return queue_.empty();
    }

    std::size_t condition_variable::size(
        std::unique_lock<mutex_type> const& lock) const
    {
        HPX_ASSERT(lock.owns_lock());
        HPX_UNUSED(lock);

        return queue_.size();
    }

    // Return false if no more threads are waiting (returns true if queue
    // is non-empty).
    bool condition_variable::notify_one(std::unique_lock<mutex_type> lock,
        threads::thread_priority /* priority */, error_code& ec)
    {
        HPX_ASSERT(lock.owns_lock());

        if (!queue_.empty())
        {
            auto ctx = queue_.front().ctx_;

            // remove item from queue before error handling
            queue_.front().ctx_.reset();
            queue_.pop_front();

            if (HPX_UNLIKELY(!ctx))
            {
                lock.unlock();

                HPX_THROWS_IF(ec, null_thread_id,
                    "condition_variable::notify_one",
                    "null thread id encountered");
                return false;
            }

            bool not_empty = !queue_.empty();
            lock.unlock();

            ctx.resume();

            return not_empty;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return false;
    }

    void condition_variable::notify_all(std::unique_lock<mutex_type> lock,
        threads::thread_priority /* priority */, error_code& ec)
    {
        HPX_ASSERT(lock.owns_lock());

        // swap the list
        queue_type queue;
        queue.swap(queue_);

        if (!queue.empty())
        {
            // update reference to queue for all queue entries
            for (queue_entry& qe : queue)
                qe.q_ = &queue;

            do
            {
                auto ctx = queue.front().ctx_;

                // remove item from queue before error handling
                queue.front().ctx_.reset();
                queue.pop_front();

                if (HPX_UNLIKELY(!ctx))
                {
                    prepend_entries(lock, queue);
                    lock.unlock();

                    HPX_THROWS_IF(ec, null_thread_id,
                        "condition_variable::notify_all",
                        "null thread id encountered");
                    return;
                }

                error_code local_ec;
                {
                    util::ignore_while_checking<std::unique_lock<mutex_type>>
                        il(&lock);
                    ctx.resume();
                }

                if (local_ec)
                {
                    prepend_entries(lock, queue);
                    lock.unlock();

                    if (&ec != &throws)
                    {
                        ec = std::move(local_ec);
                    }
                    else
                    {
                        std::rethrow_exception(
                            hpx::detail::access_exception(local_ec));
                    }
                    return;
                }

            } while (!queue.empty());
        }

        if (&ec != &throws)
            ec = make_success_code();
    }

    void condition_variable::abort_all(std::unique_lock<mutex_type> lock)
    {
        HPX_ASSERT(lock.owns_lock());

        abort_all<mutex_type>(std::move(lock));
    }

    threads::thread_restart_state condition_variable::wait(
        std::unique_lock<mutex_type>& lock, char const* /* description */,
        error_code& /* ec */)
    {
        HPX_ASSERT(lock.owns_lock());

        // enqueue the request and block this thread
        auto this_ctx = hpx::execution_base::this_thread::agent();
        queue_entry f(this_ctx, &queue_);
        queue_.push_back(f);

        reset_queue_entry r(f, queue_);
        {
            // suspend this thread
            util::unlock_guard<std::unique_lock<mutex_type>> ul(lock);
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
        HPX_ASSERT(lock.owns_lock());

        // enqueue the request and block this thread
        auto this_ctx = hpx::execution_base::this_thread::agent();
        queue_entry f(this_ctx, &queue_);
        queue_.push_back(f);

        reset_queue_entry r(f, queue_);
        {
            // suspend this thread
            util::unlock_guard<std::unique_lock<mutex_type>> ul(lock);
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
            for (queue_entry& qe : queue)
                qe.q_ = &queue;

            while (!queue.empty())
            {
                auto ctx = queue.front().ctx_;

                // remove item from queue before error handling
                queue.front().ctx_.reset();
                queue.pop_front();

                if (HPX_UNLIKELY(!ctx))
                {
                    LERR_(fatal) << "condition_variable::abort_all:"
                                 << " null thread id encountered";
                    continue;
                }

                LERR_(fatal) << "condition_variable::abort_all:"
                             << " pending thread: " << ctx;

                // unlock while notifying thread as this can suspend
                util::unlock_guard<std::unique_lock<Mutex>> unlock(lock);

                // forcefully abort thread, do not throw
                ctx.abort();
            }
        }
    }

    // re-add the remaining items to the original queue
    void condition_variable::prepend_entries(
        std::unique_lock<mutex_type>& lock, queue_type& queue)
    {
        HPX_ASSERT(lock.owns_lock());
        HPX_UNUSED(lock);

        // splice is constant time only if it == end
        queue.splice(queue.end(), queue_);
        queue_.swap(queue);
    }

    ///////////////////////////////////////////////////////////////////////////
    void intrusive_ptr_add_ref(condition_variable_data* p)
    {
        ++p->count_;
    }

    void intrusive_ptr_release(condition_variable_data* p)
    {
        if (0 == --p->count_)
        {
            delete p;
        }
    }

}}}}    // namespace hpx::lcos::local::detail

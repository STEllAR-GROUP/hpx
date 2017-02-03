//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2013-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/lcos/local/detail/condition_variable.hpp>

#include <hpx/error_code.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/lcos/local/no_mutex.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/steady_clock.hpp>
#include <hpx/util/unlock_guard.hpp>

#include <boost/exception_ptr.hpp>
#include <boost/intrusive/slist.hpp>

#include <cstddef>
#include <mutex>
#include <utility>

namespace hpx { namespace lcos { namespace local { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    condition_variable::condition_variable()
    {}

    condition_variable::~condition_variable()
    {
        if (!queue_.empty())
        {
            LERR_(fatal)
                << "~condition_variable: queue is not empty, "
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

        return queue_.empty();
    }

    std::size_t condition_variable::size(
        std::unique_lock<mutex_type> const& lock) const
    {
        HPX_ASSERT(lock.owns_lock());

        return queue_.size();
    }

    // Return false if no more threads are waiting (returns true if queue
    // is non-empty).
    bool condition_variable::notify_one(
        std::unique_lock<mutex_type> lock, threads::thread_priority priority,
        error_code& ec)
    {
        HPX_ASSERT(lock.owns_lock());

        if (!queue_.empty())
        {
            threads::thread_id_repr_type id = queue_.front().id_;

            // remove item from queue before error handling
            queue_.front().id_ = threads::invalid_thread_id_repr;
            queue_.pop_front();

            if (HPX_UNLIKELY(id == threads::invalid_thread_id_repr))
            {
                lock.unlock();

                HPX_THROWS_IF(ec, null_thread_id,
                    "condition_variable::notify_one",
                    "null thread id encountered");
                return false;
            }

            bool not_empty = !queue_.empty();
            lock.unlock();

            threads::set_thread_state(threads::thread_id_type(
                    reinterpret_cast<threads::thread_data*>(id)),
                threads::pending, threads::wait_signaled, priority, ec);

            return not_empty;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return false;
    }

    void condition_variable::notify_all(
        std::unique_lock<mutex_type> lock, threads::thread_priority priority,
        error_code& ec)
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

            do {
                threads::thread_id_repr_type id = queue.front().id_;

                // remove item from queue before error handling
                queue.front().id_ = threads::invalid_thread_id_repr;
                queue.pop_front();

                if (HPX_UNLIKELY(id == threads::invalid_thread_id_repr))
                {
                    prepend_entries(lock, queue);
                    lock.unlock();

                    HPX_THROWS_IF(ec, null_thread_id,
                        "condition_variable::notify_all",
                        "null thread id encountered");
                    return;
                }

                error_code local_ec;
                threads::set_thread_state(threads::thread_id_type(
                        reinterpret_cast<threads::thread_data*>(id)),
                    threads::pending, threads::wait_signaled,
                    priority, local_ec);

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
                        boost::rethrow_exception(
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

    threads::thread_state_ex_enum condition_variable::wait(
        std::unique_lock<mutex_type>& lock,
        char const* description, error_code& ec)
    {
        HPX_ASSERT(threads::get_self_ptr() != nullptr);
        HPX_ASSERT(lock.owns_lock());

        // enqueue the request and block this thread
        queue_entry f(threads::get_self_id().get(), &queue_);
        queue_.push_back(f);

        reset_queue_entry r(f, queue_);
        threads::thread_state_ex_enum reason = threads::wait_unknown;
        {
            // yield this thread
            util::unlock_guard<std::unique_lock<mutex_type> > ul(lock);
            reason = this_thread::suspend(threads::suspended, description, ec);
            if (ec) return threads::wait_unknown;
        }

        return (f.id_ != threads::invalid_thread_id_repr) ?
            threads::wait_timeout : reason;
    }

    threads::thread_state_ex_enum condition_variable::wait_until(
        std::unique_lock<mutex_type>& lock,
        util::steady_time_point const& abs_time,
        char const* description, error_code& ec)
    {
        HPX_ASSERT(threads::get_self_ptr() != nullptr);
        HPX_ASSERT(lock.owns_lock());

        // enqueue the request and block this thread
        queue_entry f(threads::get_self_id().get(), &queue_);
        queue_.push_back(f);

        reset_queue_entry r(f, queue_);
        threads::thread_state_ex_enum reason = threads::wait_unknown;
        {
            // yield this thread
            util::unlock_guard<std::unique_lock<mutex_type> > ul(lock);
            reason = this_thread::suspend(abs_time, description, ec);
            if (ec) return threads::wait_unknown;
        }

        return (f.id_ != threads::invalid_thread_id_repr) ?
            threads::wait_timeout : reason;
    }

    template <typename Mutex>
    void condition_variable::abort_all(std::unique_lock<Mutex> lock)
    {
        // new threads might have been added while we were notifying
        while(!queue_.empty())
        {
            // swap the list
            queue_type queue;
            queue.swap(queue_);

            // update reference to queue for all queue entries
            for (queue_entry& qe : queue)
                qe.q_ = &queue;

            while (!queue.empty())
            {
                threads::thread_id_repr_type id = queue.front().id_;

                queue.front().id_ = threads::invalid_thread_id_repr;
                queue.pop_front();

                if (HPX_UNLIKELY(id == threads::invalid_thread_id_repr))
                {
                    LERR_(fatal)
                        << "condition_variable::abort_all:"
                        << " null thread id encountered";
                    continue;
                }

                // we know that the id is actually the pointer to the thread
                threads::thread_id_type tid(
                    reinterpret_cast<threads::thread_data*>(id));

                LERR_(fatal)
                        << "condition_variable::abort_all:"
                        << " pending thread: "
                        << get_thread_state_name(
                                threads::get_thread_state(tid))
                        << "(" << tid << "): "
                        << threads::get_thread_description(tid);

                // unlock while notifying thread as this can suspend
                util::unlock_guard<std::unique_lock<Mutex> > unlock(lock);

                // forcefully abort thread, do not throw
                error_code ec(lightweight);
                threads::set_thread_state(tid,
                    threads::pending, threads::wait_abort,
                    threads::thread_priority_default, ec);
                if (ec)
                {
                    LERR_(fatal)
                        << "condition_variable::abort_all:"
                        << " could not abort thread: "
                        << get_thread_state_name(
                                threads::get_thread_state(tid))
                        << "(" << tid << "): "
                        << threads::get_thread_description(tid);
                }
            }
        }
    }

    // re-add the remaining items to the original queue
    void condition_variable::prepend_entries(
        std::unique_lock<mutex_type>& lock, queue_type& queue)
    {
        HPX_ASSERT(lock.owns_lock());

        // splice is constant time only if it == end
        queue.splice(queue.end(), queue_);
        queue_.swap(queue);
    }
}}}}

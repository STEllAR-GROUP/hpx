//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Part of this code has been adopted from code published under the BSL by:
//
//  (C) Copyright 2005-7 Anthony Williams
//  (C) Copyright 2007 David Deakins
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/lcos/local/mutex.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local
{
    mutex::~mutex()
    {
        HPX_ITT_SYNC_DESTROY(this);
        if (!queue_.empty()) {
            LERR_(fatal) << "lcos::local::mutex::~mutex: " << description_
                         << ": queue is not empty";

            mutex_type::scoped_lock l(mtx_);
            while (!queue_.empty()) {
                threads::thread_id_type id = queue_.front().id_;
                queue_.front().id_ = 0;
                queue_.pop_front();

                // we know that the id is actually the pointer to the thread
                LERR_(fatal) << "lcos::local::mutex::~mutex: " << description_
                    << ": pending thread: "
                    << threads::get_thread_state_name(threads::get_thread_state(id))
                    << "(" << id << "): " << threads::get_thread_description(id);

                // forcefully abort thread, do not throw
                error_code ec;
                threads::set_thread_state(id, threads::pending,
                    threads::wait_abort, threads::thread_priority_normal, ec);
                if (ec) {
                    LERR_(fatal) << "lcos::local::mutex::~mutex: could not abort thread"
                        << get_thread_state_name(threads::get_thread_state(id))
                        << "(" << id << "): " << threads::get_thread_state(id);
                }
            }
        }
    }

    bool mutex::wait_for_single_object(::boost::system_time const& wait_until)
    {
        // enqueue this thread
        mutex_type::scoped_lock l(mtx_);
        if (pending_events_) {
            --pending_events_;
            return false;
        }

        threads::thread_self& self = threads::get_self();
        queue_entry e(self.get_thread_id());
        queue_.push_back(e);

        bool result = false;
        threads::thread_state_ex_enum statex;

        reset_queue_entry r(e, queue_);

        {
            util::unlock_the_lock<mutex_type::scoped_lock> ul(l);

            // timeout at the given time, if appropriate
            if (!wait_until.is_not_a_date_time())
                threads::set_thread_state(self.get_thread_id(), wait_until);

            // if this timed out, return true
            statex = threads::this_thread::suspend(threads::suspended,
                description_);
        }

        return (threads::wait_timeout == statex) ? true : false;
    }

    void mutex::set_event()
    {
        if (!queue_.empty()) {
            threads::thread_id_type id = queue_.front().id_;
            queue_.front().id_ = 0;
            queue_.pop_front();

            threads::set_thread_lco_description(id);
            threads::set_thread_state(id, threads::pending);
        }
        else if (active_count_.load(boost::memory_order_acquire) & ~lock_flag_value) {
            ++pending_events_;
        }
    }

    bool mutex::timed_lock(::boost::system_time const& wait_until)
    {
        HPX_ITT_SYNC_PREPARE(this);
        if (try_lock_internal()) {
            HPX_ITT_SYNC_ACQUIRED(this);
            return true;
        }

        boost::uint32_t old_count =
            active_count_.load(boost::memory_order_acquire);
        mark_waiting_and_try_lock(old_count);

        if (old_count & lock_flag_value)
        {
            // wait for lock to get available
            bool lock_acquired = false;
            do {
                if (wait_for_single_object(wait_until))
                {
                    // if this timed out, just return false
                    --active_count_;
                    HPX_ITT_SYNC_CANCEL(this);
                    return false;
                }
                clear_waiting_and_try_lock(old_count);
                lock_acquired = !(old_count & lock_flag_value);
            } while (!lock_acquired);
        }
        HPX_ITT_SYNC_ACQUIRED(this);
        return true;
    }
}}}


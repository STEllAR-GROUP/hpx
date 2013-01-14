//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_DETAIL_SET_THREAD_STATE_JAN_13_2013_0518PM)
#define HPX_RUNTIME_THREADS_DETAIL_SET_THREAD_STATE_JAN_13_2013_0518PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/detail/create_work.hpp>
#include <hpx/util/logging.hpp>

namespace hpx { namespace threads { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy>
    thread_state set_thread_state(SchedulingPolicy& scheduler, 
        thread_id_type id, thread_state_enum new_state, 
        thread_state_ex_enum new_state_ex, thread_priority priority, 
        std::size_t thread_num, error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy>
    thread_state_enum set_active_state(SchedulingPolicy& scheduler,
        thread_id_type id, thread_state_enum newstate, 
        thread_state_ex_enum newstate_ex, thread_priority priority, 
        thread_state previous_state)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threads::detail::set_active_state",
                "NULL thread id encountered");
            return terminated;
        }

        // make sure that the thread has not been suspended and set active again
        // in the mean time
        thread_data* thrd = reinterpret_cast<thread_data*>(id);
        thread_state current_state = thrd->get_state();

        if (thread_state_enum(current_state) == thread_state_enum(previous_state) &&
            current_state != previous_state)
        {
            LTM_(warning)
                << "set_active_state: thread is still active, however "
                      "it was non-active since the original set_state "
                      "request was issued, aborting state change, thread("
                << id << "), description("
                << thrd->get_description() << "), new state("
                << get_thread_state_name(newstate) << ")";
            return terminated;
        }

        // just retry, set_state will create new thread if target is still active
        error_code ec(lightweight);      // do not throw
        set_thread_state(id, newstate, newstate_ex, priority, ec);
        return terminated;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy>
    thread_state set_thread_state(SchedulingPolicy& scheduler, 
        thread_id_type id, thread_state_enum new_state, 
        thread_state_ex_enum new_state_ex, thread_priority priority, 
        std::size_t thread_num, error_code& ec)
    {
        if (HPX_UNLIKELY(!id)) {
            HPX_THROWS_IF(ec, null_thread_id, "threads::detail::set_thread_state",
                "NULL thread id encountered");
            return thread_state(unknown);
        }

        // set_state can't be used to force a thread into active state
        if (new_state == threads::active) {
            hpx::util::osstream strm;
            strm << "invalid new state: " << get_thread_state_name(new_state);
            HPX_THROWS_IF(ec, bad_parameter, "threads::detail::set_thread_state",
                hpx::util::osstream_get_string(strm));
            return thread_state(unknown);
        }

        // we know that the id is actually the pointer to the thread
        thread_data* thrd = reinterpret_cast<thread_data*>(id);
        if (!thrd) {
            if (&ec != &throws)
                ec = make_success_code();
            return thread_state(terminated);     // this thread has already been terminated
        }

        thread_state previous_state;
        do {
            // action depends on the current state
            previous_state = thrd->get_state();
            thread_state_enum previous_state_val = previous_state;

            // nothing to do here if the state doesn't change
            if (new_state == previous_state_val) {
                LTM_(warning) 
                    << "set_thread_state: old thread state is the same as new "
                       "thread state, aborting state change, thread("
                    << id << "), description("
                    << thrd->get_description() << "), new state("
                    << get_thread_state_name(new_state) << ")";

                if (&ec != &throws)
                    ec = make_success_code();

                return thread_state(new_state);
            }

            // the thread to set the state for is currently running, so we
            // schedule another thread to execute the pending set_state
            if (active == previous_state_val) {
                // schedule a new thread to set the state
                LTM_(warning)
                    << "set_thread_state: thread is currently active, scheduling "
                        "new thread, thread(" << id << "), description("
                    << thrd->get_description() << "), new state("
                    << get_thread_state_name(new_state) << ")";

                thread_init_data data(
                    boost::bind(&set_active_state<SchedulingPolicy>, 
                        boost::ref(scheduler), id, new_state, new_state_ex, 
                        priority, previous_state),
                    "set state for active thread", 0, priority);

                create_work(scheduler, data, pending, ec);

                if (&ec != &throws)
                    ec = make_success_code();

                return previous_state;     // done
            }
            else if (terminated == previous_state_val) {
                LTM_(warning)
                    << "set_thread_state: thread is terminated, aborting state "
                        "change, thread(" << id << "), description("
                    << thrd->get_description() << "), new state("
                    << get_thread_state_name(new_state) << ")";

                if (&ec != &throws)
                    ec = make_success_code();

                // If the thread has been terminated while this set_state was
                // pending nothing has to be done anymore.
                return previous_state;
            }
            else if (pending == previous_state_val && suspended == new_state) {
                // we do not allow explicit resetting of a state to suspended
                // without the thread being executed.
                hpx::util::osstream strm;
                strm << "set_thread_state: invalid new state, can't demote a "
                        "pending thread, "
                     << "thread(" << id << "), description("
                     << thrd->get_description() << "), new state("
                     << get_thread_state_name(new_state) << ")";

                LTM_(fatal) << hpx::util::osstream_get_string(strm);

                HPX_THROWS_IF(ec, bad_parameter, 
                    "threads::detail::set_thread_state",
                    hpx::util::osstream_get_string(strm));
                return thread_state(unknown);
            }

            // If the previous state was pending we are supposed to remove the
            // thread from the queue. But in order to avoid linearly looking
            // through the queue we defer this to the thread function, which
            // at some point will ignore this thread by simply skipping it
            // (if it's not pending anymore).

            LTM_(info) << "set_thread_state: thread(" << id << "), "
                          "description(" << thrd->get_description() << "), "
                          "new state(" << get_thread_state_name(new_state) << "), "
                          "old state(" << get_thread_state_name(previous_state_val)
                       << ")";

            // So all what we do here is to set the new state.
            if (thrd->restore_state(new_state, previous_state)) {
                thrd->set_state_ex(new_state_ex);
                break;
            }

            // state has changed since we fetched it from the thread, retry
            LTM_(error) 
                << "set_thread_state: state has been changed since it was fetched, "
                   "retrying, thread(" << id << "), "
                   "description(" << thrd->get_description() << "), "
                   "new state(" << get_thread_state_name(new_state) << "), "
                   "old state(" << get_thread_state_name(previous_state_val)
                << ")";
        } while (true);

        if (new_state == pending) {
            // REVIEW: Passing a specific target thread may interfere with the
            // round robin queuing.
            scheduler.schedule_thread(thrd, thread_num, priority);
            scheduler.do_some_work();
        }

        if (&ec != &throws)
            ec = make_success_code();

        return previous_state;
    }
}}}

#endif

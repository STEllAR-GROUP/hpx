//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_DETAIL_SET_THREAD_STATE_JAN_13_2013_0518PM)
#define HPX_RUNTIME_THREADS_DETAIL_SET_THREAD_STATE_JAN_13_2013_0518PM

#include <hpx/config.hpp>
#include <hpx/config/asio.hpp>
#include <hpx/error_code.hpp>
#include <hpx/runtime/threads/coroutines/coroutine.hpp>
#include <hpx/runtime/threads/detail/create_thread.hpp>
#include <hpx/runtime/threads/detail/create_work.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind_front.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/steady_clock.hpp>

#include <boost/asio/basic_waitable_timer.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <sstream>

namespace hpx { namespace threads { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    inline thread_state set_thread_state(
        thread_id_type const& id, thread_state_enum new_state,
        thread_state_ex_enum new_state_ex, thread_priority priority,
        thread_schedule_hint schedulehint = thread_schedule_hint_none,
        error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    inline thread_result_type set_active_state(
        thread_id_type const& thrd, thread_state_enum newstate,
        thread_state_ex_enum newstate_ex, thread_priority priority,
        thread_state previous_state)
    {
        if (HPX_UNLIKELY(!thrd)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threads::detail::set_active_state",
                "null thread id encountered");
            return thread_result_type(terminated, invalid_thread_id);
        }

        // make sure that the thread has not been suspended and set active again
        // in the meantime
        thread_state current_state = thrd->get_state();

        if (current_state.state() == previous_state.state() &&
            current_state != previous_state)
        {
            LTM_(warning)
                << "set_active_state: thread is still active, however "
                      "it was non-active since the original set_state "
                      "request was issued, aborting state change, thread("
                << thrd << "), description("
                << thrd->get_description() << "), new state("
                << get_thread_state_name(newstate) << ")";
            return thread_result_type(terminated, invalid_thread_id);
        }

        // just retry, set_state will create new thread if target is still active
        error_code ec(lightweight);      // do not throw
        detail::set_thread_state(thrd, newstate, newstate_ex, priority,
            thread_schedule_hint_none, ec);

        return thread_result_type(terminated, invalid_thread_id);
    }

    ///////////////////////////////////////////////////////////////////////////
    inline thread_state set_thread_state(
        thread_id_type const& thrd, thread_state_enum new_state,
        thread_state_ex_enum new_state_ex, thread_priority priority,
        thread_schedule_hint schedulehint, error_code& ec)
    {
        if (HPX_UNLIKELY(!thrd)) {
            HPX_THROWS_IF(ec, null_thread_id, "threads::detail::set_thread_state",
                "null thread id encountered");
            return thread_state(unknown, wait_unknown);
        }

        // set_state can't be used to force a thread into active state
        if (new_state == threads::active) {
            std::ostringstream strm;
            strm << "invalid new state: " << get_thread_state_name(new_state);
            HPX_THROWS_IF(ec, bad_parameter, "threads::detail::set_thread_state",
                strm.str());
            return thread_state(unknown, wait_unknown);
        }

        // we know that the id is actually the pointer to the thread
        if (!thrd) {
            if (&ec != &throws)
                ec = make_success_code();
            return thread_state(terminated, wait_unknown);
            // this thread has already been terminated
        }

        thread_state previous_state;
        do {
            // action depends on the current state
            previous_state = thrd->get_state();
            thread_state_enum previous_state_val = previous_state.state();

            // nothing to do here if the state doesn't change
            if (new_state == previous_state_val) {
                LTM_(warning)
                    << "set_thread_state: old thread state is the same as new "
                       "thread state, aborting state change, thread("
                    << thrd << "), description("
                    << thrd->get_description() << "), new state("
                    << get_thread_state_name(new_state) << ")";

                if (&ec != &throws)
                    ec = make_success_code();

                return thread_state(new_state, previous_state.state_ex());
            }

            // the thread to set the state for is currently running, so we
            // schedule another thread to execute the pending set_state
            switch (previous_state_val) {
            case active:
                {
                    // schedule a new thread to set the state
                    LTM_(warning)
                        << "set_thread_state: thread is currently active, scheduling "
                            "new thread, thread(" << thrd << "), description("
                        << thrd->get_description() << "), new state("
                        << get_thread_state_name(new_state) << ")";

                    thread_init_data data(
                        util::bind(&set_active_state,
                            thrd, new_state, new_state_ex,
                            priority, previous_state),
                        "set state for active thread", 0, priority);

                    create_work(thrd->get_scheduler_base(), data, pending, ec);

                    if (&ec != &throws)
                        ec = make_success_code();

                    return previous_state;     // done
                }
                break;
            case terminated:
                {
                    LTM_(warning)
                        << "set_thread_state: thread is terminated, aborting state "
                            "change, thread(" << thrd << "), description("
                        << thrd->get_description() << "), new state("
                        << get_thread_state_name(new_state) << ")";

                    if (&ec != &throws)
                        ec = make_success_code();

                    // If the thread has been terminated while this set_state was
                    // pending nothing has to be done anymore.
                    return previous_state;
                }
                break;
            case pending:
            case pending_boost:
                if (suspended == new_state) {
                    // we do not allow explicit resetting of a state to suspended
                    // without the thread being executed.
                    std::ostringstream strm;
                    strm << "set_thread_state: invalid new state, can't demote a "
                            "pending thread, "
                         << "thread(" << thrd << "), description("
                         << thrd->get_description() << "), new state("
                         << get_thread_state_name(new_state) << ")";

                    LTM_(fatal) << strm.str();

                    HPX_THROWS_IF(ec, bad_parameter,
                        "threads::detail::set_thread_state",
                        strm.str());
                    return thread_state(unknown, wait_unknown);
                }
                break;
            case suspended:
                break;      // fine, just set the new state
            case pending_do_not_schedule:
            default:
                HPX_ASSERT(false);    // should not happen
                break;
            }

            // If the previous state was pending we are supposed to remove the
            // thread from the queue. But in order to avoid linearly looking
            // through the queue we defer this to the thread function, which
            // at some point will ignore this thread by simply skipping it
            // (if it's not pending anymore).

            LTM_(info) << "set_thread_state: thread(" << thrd << "), "
                          "description(" << thrd->get_description() << "), "
                          "new state(" << get_thread_state_name(new_state) << "), "
                          "old state(" << get_thread_state_name(previous_state_val)
                       << ")";

            // So all what we do here is to set the new state.
            if (thrd->restore_state(new_state, new_state_ex, previous_state))
                break;

            // state has changed since we fetched it from the thread, retry
            LTM_(error)
                << "set_thread_state: state has been changed since it was fetched, "
                   "retrying, thread(" << thrd << "), "
                   "description(" << thrd->get_description() << "), "
                   "new state(" << get_thread_state_name(new_state) << "), "
                   "old state(" << get_thread_state_name(previous_state_val)
                << ")";
        } while (true);

        if (new_state == pending) {
            // REVIEW: Passing a specific target thread may interfere with the
            // round robin queuing.
            thrd->get_scheduler_base()->schedule_thread(thrd.get(),
                schedulehint, thrd.get()->get_priority());
            thrd->get_scheduler_base()->do_some_work(schedulehint);
        }

        if (&ec != &throws)
            ec = make_success_code();

        return previous_state;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// This thread function is used by the at_timer thread below to trigger
    /// the required action.
    inline thread_result_type wake_timer_thread(
        thread_id_type const& thrd, thread_state_enum newstate,
        thread_state_ex_enum newstate_ex, thread_priority priority,
        thread_id_type const& timer_id,
        std::shared_ptr<std::atomic<bool> > const& triggered,
        thread_state_ex_enum my_statex)
    {
        if (HPX_UNLIKELY(!thrd)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threads::detail::wake_timer_thread",
                "null thread id encountered (id)");
            return thread_result_type(terminated, invalid_thread_id);
        }
        if (HPX_UNLIKELY(!timer_id)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threads::detail::wake_timer_thread",
                "null thread id encountered (timer_id)");
            return thread_result_type(terminated, invalid_thread_id);
        }

        HPX_ASSERT(my_statex == wait_abort || my_statex == wait_timeout);

        if (!triggered->load())
        {
            error_code ec(lightweight);    // do not throw
            detail::set_thread_state(timer_id, pending, my_statex,
                thread_priority_boost, thread_schedule_hint_none, ec);
        }

        return thread_result_type(terminated, invalid_thread_id);
    }

    /// This thread function initiates the required set_state action (on
    /// behalf of one of the threads#detail#set_thread_state functions).
    template <typename SchedulingPolicy>
    thread_result_type at_timer(SchedulingPolicy& scheduler,
        util::steady_clock::time_point& abs_time,
        thread_id_type const& thrd, thread_state_enum newstate,
        thread_state_ex_enum newstate_ex, thread_priority priority,
        std::atomic<bool>* started)
    {
        if (HPX_UNLIKELY(!thrd)) {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threads::detail::at_timer",
                "null thread id encountered");
            return thread_result_type(terminated, invalid_thread_id);
        }

        // create a new thread in suspended state, which will execute the
        // requested set_state when timer fires and will re-awaken this thread,
        // allowing the deadline_timer to go out of scope gracefully
        thread_id_type self_id = get_self_id();

        std::shared_ptr<std::atomic<bool> > triggered(
            std::make_shared<std::atomic<bool> >(false));

        thread_init_data data(
            util::bind_front(&wake_timer_thread,
                thrd, newstate, newstate_ex, priority,
                self_id, triggered),
            "wake_timer", 0, priority);

        thread_id_type wake_id = invalid_thread_id;
        create_thread(&scheduler, data, wake_id, suspended);

        // create timer firing in correspondence with given time
        typedef boost::asio::basic_waitable_timer<
            util::steady_clock> deadline_timer;

        deadline_timer t (
            get_thread_pool("timer-pool")->get_io_service(), abs_time);

        // let the timer invoke the set_state on the new (suspended) thread
        t.async_wait(
            [wake_id, priority](const boost::system::error_code& ec)
            {
                if (ec.value() == boost::system::errc::operation_canceled)
                {
                    detail::set_thread_state(wake_id, pending, wait_abort,
                        priority, thread_schedule_hint_none, throws);
                }
                else
                {
                    detail::set_thread_state(wake_id, pending, wait_timeout,
                        priority, thread_schedule_hint_none, throws);
                }
            });

        if (started != nullptr)
            started->store(true);

        // this waits for the thread to be reactivated when the timer fired
        // if it returns signaled the timer has been canceled, otherwise
        // the timer fired and the wake_timer_thread above has been executed
        thread_state_ex_enum statex =
            get_self().yield(thread_result_type(suspended, invalid_thread_id));

        HPX_ASSERT(statex == wait_abort || statex == wait_timeout);

        if (wait_timeout != statex) //-V601
        {
            triggered->store(true);
            // wake_timer_thread has not been executed yet, cancel timer
            t.cancel();
        }
        else
        {
            detail::set_thread_state(thrd, newstate, newstate_ex, priority);
        }

        return thread_result_type(terminated, invalid_thread_id);
    }

    /// Set a timer to set the state of the given \a thread to the given
    /// new value after it expired (at the given time)
    template <typename SchedulingPolicy>
    thread_id_type set_thread_state_timed(SchedulingPolicy& scheduler,
        util::steady_time_point const& abs_time, thread_id_type const& thrd,
        thread_state_enum newstate, thread_state_ex_enum newstate_ex,
        thread_priority priority, thread_schedule_hint schedulehint,
        std::atomic<bool>* started, error_code& ec)
    {
        if (HPX_UNLIKELY(!thrd)) {
            HPX_THROWS_IF(ec, null_thread_id,
                "threads::detail::set_thread_state",
                "null thread id encountered");
            return invalid_thread_id;
        }

        // this creates a new thread which creates the timer and handles the
        // requested actions
        thread_init_data data(
            util::bind(&at_timer<SchedulingPolicy>,
                std::ref(scheduler), abs_time.value(), thrd, newstate, newstate_ex,
                priority, started),
                "at_timer (expire at)", 0, priority, schedulehint);

        thread_id_type newid = invalid_thread_id;
        create_thread(&scheduler, data, newid, pending, ec); //-V601
        return newid;
    }

    template <typename SchedulingPolicy>
    thread_id_type set_thread_state_timed(SchedulingPolicy& scheduler,
        util::steady_time_point const& abs_time, thread_id_type const& id,
        std::atomic<bool>* started, error_code& ec)
    {
        return set_thread_state_timed(scheduler, abs_time, id, pending,
            wait_timeout, thread_priority_normal,
            thread_schedule_hint_none, started, ec);
    }

    /// Set a timer to set the state of the given \a thread to the given
    /// new value after it expired (after the given duration)
    template <typename SchedulingPolicy>
    thread_id_type set_thread_state_timed(SchedulingPolicy& scheduler,
        util::steady_duration const& rel_time, thread_id_type const& thrd,
        thread_state_enum newstate, thread_state_ex_enum newstate_ex,
        thread_priority priority, thread_schedule_hint schedulehint,
        std::atomic<bool>& started, error_code& ec)
    {
        return set_thread_state_timed(scheduler, rel_time.from_now(), thrd,
            newstate, newstate_ex, priority, schedulehint, started, ec);
    }

    template <typename SchedulingPolicy>
    thread_id_type set_thread_state_timed(SchedulingPolicy& scheduler,
        util::steady_duration const& rel_time, thread_id_type const& thrd,
        std::atomic<bool>* started, error_code& ec)
    {
        return set_thread_state_timed(scheduler, rel_time.from_now(), thrd,
            pending, wait_timeout, thread_priority_normal, thread_schedule_hint_none, started, ec);
    }
}}}

#endif

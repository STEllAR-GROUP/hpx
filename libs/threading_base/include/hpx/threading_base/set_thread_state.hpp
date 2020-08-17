//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/config/asio.hpp>
#include <hpx/assert.hpp>
#include <hpx/coroutines/coroutine.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/threading_base/create_thread.hpp>
#include <hpx/threading_base/create_work.hpp>
#include <hpx/threading_base/register_thread.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <boost/asio/basic_waitable_timer.hpp>
#include <boost/asio/io_service.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <sstream>

namespace hpx { namespace threads { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    inline thread_state set_thread_state(thread_id_type const& id,
        thread_state_enum new_state, thread_state_ex_enum new_state_ex,
        thread_priority priority,
        thread_schedule_hint schedulehint = thread_schedule_hint(),
        bool retry_on_active = true, error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    inline thread_result_type set_active_state(thread_id_type const& thrd,
        thread_state_enum newstate, thread_state_ex_enum newstate_ex,
        thread_priority priority, thread_state previous_state)
    {
        if (HPX_UNLIKELY(!thrd))
        {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threads::detail::set_active_state",
                "null thread id encountered");
            return thread_result_type(terminated, invalid_thread_id);
        }

        // make sure that the thread has not been suspended and set active again
        // in the meantime
        thread_state current_state = get_thread_id_data(thrd)->get_state();

        if (current_state.state() == previous_state.state() &&
            current_state != previous_state)
        {
            // NOLINTNEXTLINE(bugprone-branch-clone)
            LTM_(warning)
                << "set_active_state: thread is still active, however "
                   "it was non-active since the original set_state "
                   "request was issued, aborting state change, thread("
                << thrd << "), description("
                << get_thread_id_data(thrd)->get_description()
                << "), new state(" << get_thread_state_name(newstate) << ")";
            return thread_result_type(terminated, invalid_thread_id);
        }

        // just retry, set_state will create new thread if target is still active
        error_code ec(lightweight);    // do not throw
        detail::set_thread_state(thrd, newstate, newstate_ex, priority,
            thread_schedule_hint(), true, ec);

        return thread_result_type(terminated, invalid_thread_id);
    }

    ///////////////////////////////////////////////////////////////////////////
    inline thread_state set_thread_state(thread_id_type const& thrd,
        thread_state_enum new_state, thread_state_ex_enum new_state_ex,
        thread_priority priority, thread_schedule_hint schedulehint,
        bool retry_on_active, error_code& ec)
    {
        if (HPX_UNLIKELY(!thrd))
        {
            HPX_THROWS_IF(ec, null_thread_id,
                "threads::detail::set_thread_state",
                "null thread id encountered");
            return thread_state(unknown, wait_unknown);
        }

        // set_state can't be used to force a thread into active state
        if (new_state == threads::active)
        {
            std::ostringstream strm;
            strm << "invalid new state: " << get_thread_state_name(new_state);
            HPX_THROWS_IF(ec, bad_parameter,
                "threads::detail::set_thread_state", strm.str());
            return thread_state(unknown, wait_unknown);
        }

        thread_state previous_state;
        do
        {
            // action depends on the current state
            previous_state = get_thread_id_data(thrd)->get_state();
            thread_state_enum previous_state_val = previous_state.state();

            // nothing to do here if the state doesn't change
            if (new_state == previous_state_val)
            {
                // NOLINTNEXTLINE(bugprone-branch-clone)
                LTM_(warning)
                    << "set_thread_state: old thread state is the same as new "
                       "thread state, aborting state change, thread("
                    << thrd << "), description("
                    << get_thread_id_data(thrd)->get_description()
                    << "), new state(" << get_thread_state_name(new_state)
                    << ")";

                if (&ec != &throws)
                    ec = make_success_code();

                return thread_state(new_state, previous_state.state_ex());
            }

            // the thread to set the state for is currently running, so we
            // schedule another thread to execute the pending set_state
            switch (previous_state_val)
            {
            case active:
            {
                if (retry_on_active)
                {
                    // schedule a new thread to set the state
                    // NOLINTNEXTLINE(bugprone-branch-clone)
                    LTM_(warning)
                        << "set_thread_state: thread is currently active, "
                           "scheduling new thread, thread("
                        << thrd << "), description("
                        << get_thread_id_data(thrd)->get_description()
                        << "), new state(" << get_thread_state_name(new_state)
                        << ")";

                    thread_init_data data(
                        util::bind(&set_active_state, thrd, new_state,
                            new_state_ex, priority, previous_state),
                        "set state for active thread", priority);

                    create_work(get_thread_id_data(thrd)->get_scheduler_base(),
                        data, ec);

                    if (&ec != &throws)
                        ec = make_success_code();
                }
                else
                {
                    // NOLINTNEXTLINE(bugprone-branch-clone)
                    LTM_(warning)
                        << "set_thread_state: thread is currently active, "
                           "but not scheduling new thread because "
                           "retry_on_active = false, thread("
                        << thrd << "), description("
                        << get_thread_id_data(thrd)->get_description()
                        << "), new state(" << get_thread_state_name(new_state)
                        << ")";
                    ec = make_success_code();
                }

                return previous_state;    // done
            }
            break;
            case terminated:
            {
                // NOLINTNEXTLINE(bugprone-branch-clone)
                LTM_(warning) << "set_thread_state: thread is terminated, "
                                 "aborting state "
                                 "change, thread("
                              << thrd << "), description("
                              << get_thread_id_data(thrd)->get_description()
                              << "), new state("
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
                if (suspended == new_state)
                {
                    // we do not allow explicit resetting of a state to suspended
                    // without the thread being executed.
                    std::ostringstream strm;
                    strm << "set_thread_state: invalid new state, can't demote "
                            "a "
                            "pending thread, "
                         << "thread(" << thrd << "), description("
                         << get_thread_id_data(thrd)->get_description()
                         << "), new state(" << get_thread_state_name(new_state)
                         << ")";

                    // NOLINTNEXTLINE(bugprone-branch-clone)
                    LTM_(fatal) << strm.str();

                    HPX_THROWS_IF(ec, bad_parameter,
                        "threads::detail::set_thread_state", strm.str());
                    return thread_state(unknown, wait_unknown);
                }
                break;
            case suspended:
                break;    // fine, just set the new state
            case pending_do_not_schedule:
                HPX_FALLTHROUGH;
            default:
            {
                std::ostringstream strm;
                strm << "set_thread_state: previous state was "
                     << get_thread_state_name(previous_state_val) << " ("
                     << previous_state_val << ")";
                HPX_ASSERT_MSG(
                    false, strm.str().c_str());    // should not happen
                break;
            }
            }

            // If the previous state was pending we are supposed to remove the
            // thread from the queue. But in order to avoid linearly looking
            // through the queue we defer this to the thread function, which
            // at some point will ignore this thread by simply skipping it
            // (if it's not pending anymore).

            // NOLINTNEXTLINE(bugprone-branch-clone)
            LTM_(info) << "set_thread_state: thread(" << thrd
                       << "), description("
                       << get_thread_id_data(thrd)->get_description()
                       << "), new state(" << get_thread_state_name(new_state)
                       << "), old state("
                       << get_thread_state_name(previous_state_val) << ")";

            // So all what we do here is to set the new state.
            if (get_thread_id_data(thrd)->restore_state(
                    new_state, new_state_ex, previous_state))
            {
                break;
            }

            // state has changed since we fetched it from the thread, retry
            // NOLINTNEXTLINE(bugprone-branch-clone)
            LTM_(error) << "set_thread_state: state has been changed since it "
                           "was fetched, "
                           "retrying, thread("
                        << thrd
                        << "), "
                           "description("
                        << get_thread_id_data(thrd)->get_description()
                        << "), "
                           "new state("
                        << get_thread_state_name(new_state)
                        << "), "
                           "old state("
                        << get_thread_state_name(previous_state_val) << ")";
        } while (true);

        thread_state_enum previous_state_val = previous_state.state();
        if (!(previous_state_val == pending ||
                previous_state_val == pending_boost) &&
            (new_state == pending || new_state == pending_boost))
        {
            // REVIEW: Passing a specific target thread may interfere with the
            // round robin queuing.

            auto* thrd_data = get_thread_id_data(thrd);
            auto* scheduler = thrd_data->get_scheduler_base();
            scheduler->schedule_thread(
                thrd_data, schedulehint, false, thrd_data->get_priority());
            // NOTE: Don't care if the hint is a NUMA hint, just want to wake up
            // a thread.
            scheduler->do_some_work(schedulehint.hint);
        }

        if (&ec != &throws)
            ec = make_success_code();

        return previous_state;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// This thread function is used by the at_timer thread below to trigger
    /// the required action.
    inline thread_result_type wake_timer_thread(thread_id_type const& thrd,
        thread_state_enum /*newstate*/, thread_state_ex_enum /*newstate_ex*/,
        thread_priority /*priority*/, thread_id_type const& timer_id,
        std::shared_ptr<std::atomic<bool>> const& triggered,
        bool retry_on_active, thread_state_ex_enum my_statex)
    {
        if (HPX_UNLIKELY(!thrd))
        {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threads::detail::wake_timer_thread",
                "null thread id encountered (id)");
            return thread_result_type(terminated, invalid_thread_id);
        }
        if (HPX_UNLIKELY(!timer_id))
        {
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
                thread_priority_boost, thread_schedule_hint(), retry_on_active,
                ec);
        }

        return thread_result_type(terminated, invalid_thread_id);
    }

    /// This thread function initiates the required set_state action (on
    /// behalf of one of the threads#detail#set_thread_state functions).
    template <typename SchedulingPolicy>
    thread_result_type at_timer(SchedulingPolicy& scheduler,
        util::steady_clock::time_point& abs_time, thread_id_type const& thrd,
        thread_state_enum newstate, thread_state_ex_enum newstate_ex,
        thread_priority priority, std::atomic<bool>* started,
        bool retry_on_active)
    {
        if (HPX_UNLIKELY(!thrd))
        {
            HPX_THROW_EXCEPTION(null_thread_id, "threads::detail::at_timer",
                "null thread id encountered");
            return thread_result_type(terminated, invalid_thread_id);
        }

        // create a new thread in suspended state, which will execute the
        // requested set_state when timer fires and will re-awaken this thread,
        // allowing the deadline_timer to go out of scope gracefully
        thread_id_type self_id = get_self_id();

        std::shared_ptr<std::atomic<bool>> triggered(
            std::make_shared<std::atomic<bool>>(false));

        thread_init_data data(
            util::bind_front(&wake_timer_thread, thrd, newstate, newstate_ex,
                priority, self_id, triggered, retry_on_active),
            "wake_timer", priority, thread_schedule_hint(),
            thread_stacksize_small, suspended, true);

        thread_id_type wake_id = invalid_thread_id;
        create_thread(&scheduler, data, wake_id);

        // create timer firing in correspondence with given time
        using deadline_timer =
            boost::asio::basic_waitable_timer<util::steady_clock>;

        boost::asio::io_service* s = get_default_timer_service();
        HPX_ASSERT(s);
        deadline_timer t(*s, abs_time);

        // let the timer invoke the set_state on the new (suspended) thread
        t.async_wait([wake_id, priority, retry_on_active](
                         const boost::system::error_code& ec) {
            if (ec.value() == boost::system::errc::operation_canceled)
            {
                detail::set_thread_state(wake_id, pending, wait_abort, priority,
                    thread_schedule_hint(), retry_on_active, throws);
            }
            else
            {
                detail::set_thread_state(wake_id, pending, wait_timeout,
                    priority, thread_schedule_hint(), retry_on_active, throws);
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

        if (wait_timeout != statex)    //-V601
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
        std::atomic<bool>* started, bool retry_on_active, error_code& ec)
    {
        if (HPX_UNLIKELY(!thrd))
        {
            HPX_THROWS_IF(ec, null_thread_id,
                "threads::detail::set_thread_state",
                "null thread id encountered");
            return invalid_thread_id;
        }

        // this creates a new thread which creates the timer and handles the
        // requested actions
        thread_init_data data(
            util::bind(&at_timer<SchedulingPolicy>, std::ref(scheduler),
                abs_time.value(), thrd, newstate, newstate_ex, priority,
                started, retry_on_active),
            "at_timer (expire at)", priority, schedulehint,
            thread_stacksize_small, pending, true);

        thread_id_type newid = invalid_thread_id;
        create_thread(&scheduler, data, newid, ec);    //-V601
        return newid;
    }

    template <typename SchedulingPolicy>
    thread_id_type set_thread_state_timed(SchedulingPolicy& scheduler,
        util::steady_time_point const& abs_time, thread_id_type const& id,
        std::atomic<bool>* started, bool retry_on_active, error_code& ec)
    {
        return set_thread_state_timed(scheduler, abs_time, id, pending,
            wait_timeout, thread_priority_normal, thread_schedule_hint(),
            started, retry_on_active, ec);
    }

    /// Set a timer to set the state of the given \a thread to the given
    /// new value after it expired (after the given duration)
    template <typename SchedulingPolicy>
    thread_id_type set_thread_state_timed(SchedulingPolicy& scheduler,
        util::steady_duration const& rel_time, thread_id_type const& thrd,
        thread_state_enum newstate, thread_state_ex_enum newstate_ex,
        thread_priority priority, thread_schedule_hint schedulehint,
        std::atomic<bool>& started, bool retry_on_active, error_code& ec)
    {
        return set_thread_state_timed(scheduler, rel_time.from_now(), thrd,
            newstate, newstate_ex, priority, schedulehint, started,
            retry_on_active, ec);
    }

    template <typename SchedulingPolicy>
    thread_id_type set_thread_state_timed(SchedulingPolicy& scheduler,
        util::steady_duration const& rel_time, thread_id_type const& thrd,
        std::atomic<bool>* started, bool retry_on_active, error_code& ec)
    {
        return set_thread_state_timed(scheduler, rel_time.from_now(), thrd,
            pending, wait_timeout, thread_priority_normal,
            thread_schedule_hint(), started, retry_on_active, ec);
    }
}}}    // namespace hpx::threads::detail

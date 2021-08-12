//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/config/asio.hpp>
#include <hpx/assert.hpp>
#include <hpx/coroutines/coroutine.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/threading_base/create_thread.hpp>
#include <hpx/threading_base/detail/get_default_timer_service.hpp>
#include <hpx/threading_base/set_thread_state_timed.hpp>
#include <hpx/threading_base/threading_base_fwd.hpp>

#include <asio/basic_waitable_timer.hpp>
#include <asio/io_context.hpp>

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <system_error>
#include <utility>

namespace hpx { namespace threads { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    /// This thread function is used by the at_timer thread below to trigger
    /// the required action.
    thread_result_type wake_timer_thread(thread_id_ref_type const& thrd,
        thread_schedule_state /*newstate*/,
        thread_restart_state /*newstate_ex*/, thread_priority /*priority*/,
        thread_id_type timer_id,
        std::shared_ptr<std::atomic<bool>> const& triggered,
        bool retry_on_active, thread_restart_state my_statex)
    {
        if (HPX_UNLIKELY(!thrd))
        {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threads::detail::wake_timer_thread",
                "null thread id encountered (id)");
            return thread_result_type(
                thread_schedule_state::terminated, invalid_thread_id);
        }

        if (HPX_UNLIKELY(!timer_id))
        {
            HPX_THROW_EXCEPTION(null_thread_id,
                "threads::detail::wake_timer_thread",
                "null thread id encountered (timer_id)");
            return thread_result_type(
                thread_schedule_state::terminated, invalid_thread_id);
        }

        HPX_ASSERT(my_statex == thread_restart_state::abort ||
            my_statex == thread_restart_state::timeout);

        if (!triggered->load())
        {
            error_code ec(lightweight);    // do not throw
            set_thread_state(timer_id, thread_schedule_state::pending,
                my_statex, thread_priority::boost, thread_schedule_hint(),
                retry_on_active, ec);
        }

        return thread_result_type(
            thread_schedule_state::terminated, invalid_thread_id);
    }

    /// This thread function initiates the required set_state action (on
    /// behalf of one of the threads#detail#set_thread_state functions).
    thread_result_type at_timer(policies::scheduler_base* scheduler,
        std::chrono::steady_clock::time_point& abs_time,
        thread_id_ref_type const& thrd, thread_schedule_state newstate,
        thread_restart_state newstate_ex, thread_priority priority,
        std::atomic<bool>* started, bool retry_on_active)
    {
        if (HPX_UNLIKELY(!thrd))
        {
            HPX_THROW_EXCEPTION(null_thread_id, "threads::detail::at_timer",
                "null thread id encountered");
            return thread_result_type(
                thread_schedule_state::terminated, invalid_thread_id);
        }

        // create a new thread in suspended state, which will execute the
        // requested set_state when timer fires and will re-awaken this thread,
        // allowing the deadline_timer to go out of scope gracefully
        thread_id_ref_type self_id = get_self_id();    // keep alive

        std::shared_ptr<std::atomic<bool>> triggered(
            std::make_shared<std::atomic<bool>>(false));

        thread_init_data data(
            util::bind_front(&wake_timer_thread, thrd, newstate, newstate_ex,
                priority, self_id.noref(), triggered, retry_on_active),
            "wake_timer", priority, thread_schedule_hint(),
            thread_stacksize::small_, thread_schedule_state::suspended, true);

        thread_id_ref_type wake_id = invalid_thread_id;
        create_thread(scheduler, data, wake_id);

        // create timer firing in correspondence with given time
        using deadline_timer =
            asio::basic_waitable_timer<std::chrono::steady_clock>;

        asio::io_context* s = get_default_timer_service();
        HPX_ASSERT(s);
        deadline_timer t(*s, abs_time);

        // let the timer invoke the set_state on the new (suspended) thread
        t.async_wait([wake_id = std::move(wake_id), priority, retry_on_active](
                         std::error_code const& ec) {
            if (ec == std::make_error_code(std::errc::operation_canceled))
            {
                set_thread_state(wake_id.noref(),
                    thread_schedule_state::pending, thread_restart_state::abort,
                    priority, thread_schedule_hint(), retry_on_active, throws);
            }
            else
            {
                set_thread_state(wake_id.noref(),
                    thread_schedule_state::pending,
                    thread_restart_state::timeout, priority,
                    thread_schedule_hint(), retry_on_active, throws);
            }
        });

        if (started != nullptr)
        {
            started->store(true);
        }

        // this waits for the thread to be reactivated when the timer fired
        // if it returns signaled the timer has been canceled, otherwise
        // the timer fired and the wake_timer_thread above has been executed
        thread_restart_state statex = get_self().yield(thread_result_type(
            thread_schedule_state::suspended, invalid_thread_id));

        HPX_ASSERT(statex == thread_restart_state::abort ||
            statex == thread_restart_state::timeout);

        // NOLINTNEXTLINE(bugprone-branch-clone)
        if (thread_restart_state::timeout != statex)    //-V601
        {
            triggered->store(true);
            // wake_timer_thread has not been executed yet, cancel timer
            t.cancel();
        }
        else
        {
            detail::set_thread_state(
                thrd.noref(), newstate, newstate_ex, priority);
        }

        return thread_result_type(
            thread_schedule_state::terminated, invalid_thread_id);
    }

    /// Set a timer to set the state of the given \a thread to the given
    /// new value after it expired (at the given time)
    thread_id_ref_type set_thread_state_timed(
        policies::scheduler_base* scheduler,
        hpx::chrono::steady_time_point const& abs_time,
        thread_id_type const& thrd, thread_schedule_state newstate,
        thread_restart_state newstate_ex, thread_priority priority,
        thread_schedule_hint schedulehint, std::atomic<bool>* started,
        bool retry_on_active, error_code& ec)
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
            util::bind(&at_timer, scheduler, abs_time.value(),
                thread_id_ref_type(thrd), newstate, newstate_ex, priority,
                started, retry_on_active),
            "at_timer (expire at)", priority, schedulehint,
            thread_stacksize::small_, thread_schedule_state::pending, true);

        thread_id_ref_type newid = invalid_thread_id;
        create_thread(scheduler, data, newid, ec);    //-V601
        return newid;
    }
}}}    // namespace hpx::threads::detail

//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/coroutines/coroutine.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/threading_base/create_work.hpp>
#include <hpx/threading_base/thread_init_data.hpp>
#include <hpx/modules/timing.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/thread_data.hpp>

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

namespace hpx { namespace threads { namespace detail {

    HPX_CORE_EXPORT thread_state set_thread_state(thread_id_type const& id,
        thread_schedule_state new_state, thread_restart_state new_state_ex,
        thread_priority priority,
        thread_schedule_hint schedulehint = thread_schedule_hint(),
        bool retry_on_active = true, error_code& ec = throws);

    HPX_CORE_EXPORT thread_result_type set_active_state(
        thread_id_type const& thrd, thread_schedule_state newstate,
        thread_restart_state newstate_ex, thread_priority priority,
        thread_state previous_state);

    HPX_CORE_EXPORT thread_state set_thread_state(thread_id_type const& thrd,
        thread_schedule_state new_state, thread_restart_state new_state_ex,
        thread_priority priority, thread_schedule_hint schedulehint,
        bool retry_on_active, error_code& ec);

    // This thread function is used by the at_timer thread below to trigger
    // the required action.
    HPX_CORE_EXPORT thread_result_type wake_timer_thread(
        thread_id_type const& thrd, thread_schedule_state newstate,
        thread_restart_state newstate_ex, thread_priority priority,
        thread_id_type timer_id,
        std::shared_ptr<std::atomic<bool>> const& triggered,
        bool retry_on_active, thread_restart_state my_statex);

    // This thread function initiates the required set_state action (on
    // behalf of one of the threads#detail#set_thread_state functions).
    HPX_CORE_EXPORT thread_result_type at_timer(
        policies::scheduler_base* scheduler,
        std::chrono::steady_clock::time_point& abs_time,
        thread_id_type const& thrd, thread_schedule_state newstate,
        thread_restart_state newstate_ex, thread_priority priority,
        std::atomic<bool>* started, bool retry_on_active);

    // Set a timer to set the state of the given \a thread to the given
    // new value after it expired (at the given time)
    HPX_CORE_EXPORT thread_id_type set_thread_state_timed(
        policies::scheduler_base* scheduler,
        hpx::chrono::steady_time_point const& abs_time,
        thread_id_type const& thrd, thread_schedule_state newstate,
        thread_restart_state newstate_ex, thread_priority priority,
        thread_schedule_hint schedulehint, std::atomic<bool>* started,
        bool retry_on_active, error_code& ec);

    inline thread_id_type set_thread_state_timed(
        policies::scheduler_base* scheduler,
        hpx::chrono::steady_time_point const& abs_time,
        thread_id_type const& id, std::atomic<bool>* started,
        bool retry_on_active, error_code& ec)
    {
        return set_thread_state_timed(scheduler, abs_time, id,
            thread_schedule_state::pending, thread_restart_state::timeout,
            thread_priority::normal, thread_schedule_hint(), started,
            retry_on_active, ec);
    }

    // Set a timer to set the state of the given \a thread to the given
    // new value after it expired (after the given duration)
    inline thread_id_type set_thread_state_timed(
        policies::scheduler_base* scheduler,
        hpx::chrono::steady_duration const& rel_time,
        thread_id_type const& thrd, thread_schedule_state newstate,
        thread_restart_state newstate_ex, thread_priority priority,
        thread_schedule_hint schedulehint, std::atomic<bool>* started,
        bool retry_on_active, error_code& ec)
    {
        return set_thread_state_timed(scheduler, rel_time.from_now(), thrd,
            newstate, newstate_ex, priority, schedulehint, started,
            retry_on_active, ec);
    }

    inline thread_id_type set_thread_state_timed(
        policies::scheduler_base* scheduler,
        hpx::chrono::steady_duration const& rel_time,
        thread_id_type const& thrd, std::atomic<bool>* started,
        bool retry_on_active, error_code& ec)
    {
        return set_thread_state_timed(scheduler, rel_time.from_now(), thrd,
            thread_schedule_state::pending, thread_restart_state::timeout,
            thread_priority::normal, thread_schedule_hint(), started,
            retry_on_active, ec);
    }
}}}    // namespace hpx::threads::detail

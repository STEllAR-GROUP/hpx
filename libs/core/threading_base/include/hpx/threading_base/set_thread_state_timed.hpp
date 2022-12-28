//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/coroutines/coroutine.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/timing.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/set_thread_state.hpp>
#include <hpx/threading_base/threading_base_fwd.hpp>

#include <atomic>

namespace hpx::threads::detail {

    /// Set a timer to set the state of the given \a thread to the given
    /// new value after it expired (at the given time)
    HPX_CORE_EXPORT thread_id_ref_type set_thread_state_timed(
        policies::scheduler_base* scheduler,
        hpx::chrono::steady_time_point const& abs_time,
        thread_id_type const& thrd, thread_schedule_state newstate,
        thread_restart_state newstate_ex, thread_priority priority,
        thread_schedule_hint schedulehint, std::atomic<bool>* started,
        bool retry_on_active, error_code& ec);

    inline thread_id_ref_type set_thread_state_timed(
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
    inline thread_id_ref_type set_thread_state_timed(
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

    inline thread_id_ref_type set_thread_state_timed(
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
}    // namespace hpx::threads::detail

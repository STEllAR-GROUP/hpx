//  Copyright (c) 2022-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_LOGGING)
#include <hpx/modules/coroutines.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/modules/threading_base.hpp>
#include <hpx/thread_pools/detail/scheduling_log.hpp>

#include <cstddef>

namespace hpx::threads::detail {

    ///////////////////////////////////////////////////////////////////////
    void write_state_log(policies::scheduler_base const& scheduler,
        std::size_t const num_thread, thread_id_ref_type const& thrd,
        thread_schedule_state const old_state,
        thread_schedule_state const new_state)
    {
        LTM_(debug).format("scheduling_loop state change: pool({}), "
                           "scheduler({}), worker_thread({}), thread({}), "
                           "description({}), old state({}), new state({})",
            *scheduler.get_parent_pool(), scheduler, num_thread,
            get_thread_id_data(thrd),
            get_thread_id_data(thrd)->get_description(),
            get_thread_state_name(old_state), get_thread_state_name(new_state));
    }

    void write_state_log_warning(policies::scheduler_base const& scheduler,
        std::size_t const num_thread, thread_id_ref_type const& thrd,
        thread_schedule_state const state, char const* info)
    {
        LTM_(warning).format("scheduling_loop state change failed: pool({}), "
                             "scheduler({}), worker thread ({}), thread({}), "
                             "description({}), state({}), {}",
            *scheduler.get_parent_pool(), scheduler, num_thread,
            get_thread_id_data(thrd)->get_thread_id(),
            get_thread_id_data(thrd)->get_description(),
            get_thread_state_name(state), info);
    }

    void write_rescheduling_log_warning(
        policies::scheduler_base const& scheduler, std::size_t const num_thread,
        thread_id_ref_type const& thrd)
    {
        LTM_(warning).format("pool({}), scheduler({}), worker_thread({}), "
                             "thread({}), description({}), rescheduling",
            *scheduler.get_parent_pool(), scheduler, num_thread,
            get_thread_id_data(thrd)->get_thread_id(),
            get_thread_id_data(thrd)->get_description());
    }
}    // namespace hpx::threads::detail

#endif

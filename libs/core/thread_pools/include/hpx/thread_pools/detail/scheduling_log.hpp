//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/threading_base_fwd.hpp>

#include <cstddef>

namespace hpx::threads::detail {

#if defined(HPX_HAVE_LOGGING)
    HPX_CORE_EXPORT void write_state_log(
        policies::scheduler_base const& scheduler, std::size_t num_thread,
        thread_id_ref_type const& thrd, thread_schedule_state const old_state,
        thread_schedule_state const new_state);

    HPX_CORE_EXPORT void write_state_log_warning(
        policies::scheduler_base const& scheduler, std::size_t num_thread,
        thread_id_ref_type const& thrd, thread_schedule_state state,
        char const* info);

    HPX_CORE_EXPORT void write_rescheduling_log_warning(
        policies::scheduler_base const& scheduler, std::size_t num_thread,
        thread_id_ref_type const& thrd);
#else
    constexpr void write_state_log(policies::scheduler_base const&, std::size_t,
        thread_id_ref_type const&, thread_schedule_state const,
        thread_schedule_state const) noexcept
    {
    }

    constexpr void write_state_log_warning(policies::scheduler_base const&,
        std::size_t, thread_id_ref_type const&, thread_schedule_state,
        char const*) noexcept
    {
    }

    constexpr void write_rescheduling_log_warning(
        policies::scheduler_base const&, std::size_t,
        thread_id_ref_type const&) noexcept
    {
    }
#endif
}    // namespace hpx::threads::detail

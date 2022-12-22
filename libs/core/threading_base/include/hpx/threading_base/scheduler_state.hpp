////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>

#include <cstdint>

namespace hpx {

    enum class state : std::int8_t
    {
        invalid = -1,
        initialized = 0,
        pre_startup = 1,
        startup = 2,
        pre_main = 3,
        starting = 4,
        running = 5,
        suspended = 6,
        pre_sleep = 7,
        sleeping = 8,
        pre_shutdown = 9,
        shutdown = 10,
        stopping = 11,
        terminating = 12,
        stopped = 13,
        first_valid_runtime_state = initialized,
        last_valid_runtime_state = stopped
    };

#define HPX_STATE_UNSCOPED_ENUM_DEPRECATION_MSG                                \
    "The unscoped state names are deprecated. Please use state::<state> "      \
    "instead."

    HPX_DEPRECATED_V(1, 8, HPX_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr state state_invalid = state::invalid;
    HPX_DEPRECATED_V(1, 8, HPX_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr state state_initialized = state::initialized;
    HPX_DEPRECATED_V(1, 8, HPX_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr state state_pre_startup = state::pre_startup;
    HPX_DEPRECATED_V(1, 8, HPX_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr state state_startup = state::startup;
    HPX_DEPRECATED_V(1, 8, HPX_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr state state_pre_main = state::pre_main;
    HPX_DEPRECATED_V(1, 8, HPX_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr state state_starting = state::starting;
    HPX_DEPRECATED_V(1, 8, HPX_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr state state_running = state::running;
    HPX_DEPRECATED_V(1, 8, HPX_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr state state_suspended = state::suspended;
    HPX_DEPRECATED_V(1, 8, HPX_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr state state_pre_sleep = state::pre_sleep;
    HPX_DEPRECATED_V(1, 8, HPX_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr state state_sleeping = state::sleeping;
    HPX_DEPRECATED_V(1, 8, HPX_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr state state_pre_shutdown = state::pre_shutdown;
    HPX_DEPRECATED_V(1, 8, HPX_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr state state_shutdown = state::shutdown;
    HPX_DEPRECATED_V(1, 8, HPX_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr state state_stopping = state::stopping;
    HPX_DEPRECATED_V(1, 8, HPX_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr state state_terminating = state::terminating;
    HPX_DEPRECATED_V(1, 8, HPX_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr state state_stopped = state::stopped;
    HPX_DEPRECATED_V(1, 8, HPX_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr state first_valid_runtime_state =
        state::first_valid_runtime_state;
    HPX_DEPRECATED_V(1, 8, HPX_STATE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr state state_last_valid_runtime_state =
        state::last_valid_runtime_state;

#undef HPX_STATE_UNSCOPED_ENUM_DEPRECATION_MSG
}    // namespace hpx

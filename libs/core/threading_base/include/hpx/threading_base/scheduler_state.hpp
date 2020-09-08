////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>

namespace hpx {
    enum state
    {
        state_invalid = -1,
        state_initialized = 0,
        first_valid_runtime_state = state_initialized,
        state_pre_startup = 1,
        state_startup = 2,
        state_pre_main = 3,
        state_starting = 4,
        state_running = 5,
        state_suspended = 6,
        state_pre_sleep = 7,
        state_sleeping = 8,
        state_pre_shutdown = 9,
        state_shutdown = 10,
        state_stopping = 11,
        state_terminating = 12,
        state_stopped = 13,
        last_valid_runtime_state = state_stopped
    };
}

////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_703646B3_0567_484E_AD34_A752B8163B30)
#define HPX_703646B3_0567_484E_AD34_A752B8163B30

#include <hpx/config.hpp>

namespace hpx
{
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
        state_pre_shutdown = 7,
        state_shutdown = 8,
        state_stopping = 9,
        state_terminating = 10,
        state_stopped = 11,
        last_valid_runtime_state = state_stopped
    };

    namespace threads
    {
        // return whether thread manager is in the state described by 'mask'
        HPX_EXPORT bool threadmanager_is(state st);
        HPX_EXPORT bool threadmanager_is_at_least(state st);
    }

    namespace agas
    {
        // return whether resolver client is in state described by 'mask'
        HPX_EXPORT bool router_is(state st);
    }
}

#endif // HPX_703646B3_0567_484E_AD34_A752B8163B30


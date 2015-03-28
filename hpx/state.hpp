////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_703646B3_0567_484E_AD34_A752B8163B30)
#define HPX_703646B3_0567_484E_AD34_A752B8163B30

#include <boost/atomic.hpp>
#include <boost/cstdint.hpp>
#include <boost/utility/binary.hpp>

#include <hpx/config.hpp>

namespace hpx
{
    enum state
    {
        state_invalid      = -1,
        initialized        = 0,
        starting           = 1,
        running            = 2,
        stopping           = 3,
        terminating        = 4,
        state_initialized  = 5,
        state_pre_startup  = 6,
        state_startup      = 7,
        state_pre_main     = 8,
        state_running      = 9,
        state_pre_shutdown = 10,
        state_shutdown     = 11,
        state_stopped      = 12,
        state_last         = state_stopped
    };

    namespace threads
    {
        // return whether thread manager is in the state described by 'mask'
        HPX_EXPORT bool threadmanager_is(boost::uint8_t mask);

        // Forwarder
        inline bool threadmanager_is(state mask)
        {
            return threadmanager_is(boost::uint8_t(mask));
        }
    }

    namespace agas
    {
        // return whether resolver client is in state described by 'mask'
        HPX_EXPORT bool router_is(boost::uint8_t mask);

        // Forwarder
        inline bool router_is(state mask)
        {
            return router_is(boost::uint8_t(mask));
        }
    }
}

#endif // HPX_703646B3_0567_484E_AD34_A752B8163B30


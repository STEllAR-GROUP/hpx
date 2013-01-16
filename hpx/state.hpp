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
        starting = BOOST_BINARY_U(001),
        running  = BOOST_BINARY_U(010),
        stopping = BOOST_BINARY_U(100)
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


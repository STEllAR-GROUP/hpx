////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_fwd.hpp>
#include <hpx/state.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    namespace threads
    {
        // return whether thread manager is in the state described by 'mask'
        bool threadmanager_is(boost::uint8_t mask)
        {
            hpx::runtime* rt = get_runtime_ptr();
            if (NULL == rt) {
                // we're probably either starting or stopping
                return (mask & (starting | stopping)) != 0;
            }
            return (rt->get_thread_manager().status() & mask) != 0;
        }
    }

    namespace agas
    {
        // return whether resolver client is in state described by 'mask'
        bool router_is(boost::uint8_t mask)
        {
            runtime* rt = get_runtime_ptr();
            if (NULL == rt) {
                // we're probably either starting or stopping
                return (mask & (starting | stopping)) != 0;
            }
            return (rt->get_agas_client().get_status() & mask) != 0;
        }
    }
}


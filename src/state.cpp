////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_fwd.hpp>
#include <hpx/state.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>

namespace hpx
{

namespace threads
{

bool threadmanager_is(boost::uint8_t mask)
{
    if (NULL == applier::get_applier_ptr())
        return false; 
    return (applier::get_applier_ptr()->get_thread_manager().status() & mask) ? true : false;
}

}

#if HPX_AGAS_VERSION > 0x10
    namespace agas
    {

    bool router_is(boost::uint8_t mask)
    {
        if (NULL == get_runtime_ptr()) 
            return false;
        return (get_runtime_ptr()->get_agas_client().status() & mask) ? true : false;
    }

    } 
#endif

}


////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/state.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    namespace threads
    {
        // return whether thread manager is in the state described by st
        bool threadmanager_is(state st)
        {
            hpx::runtime* rt = get_runtime_ptr();
            if (nullptr == rt) {
                // we're probably either starting or stopping
                return st == state_starting || st == state_stopping;
            }
            return (rt->get_thread_manager().status() == st);
        }
    }

    namespace agas
    {
        // return whether resolver client is in state described by st
        bool router_is(state st)
        {
            runtime* rt = get_runtime_ptr();
            if (nullptr == rt) {
                // we're probably either starting or stopping
                return st == state_starting || st == state_stopping;
            }
            return (rt->get_agas_client().get_status() == st);
        }
    }
}


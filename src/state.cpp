////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/state.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
#include <hpx/runtime_distributed.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#endif

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
                return st <= state_starting || st >= state_stopping;
            }
            return (rt->get_thread_manager().status() == st);
        }
        bool threadmanager_is_at_least(state st)
        {
            hpx::runtime* rt = get_runtime_ptr();
            if (nullptr == rt) {
                // we're probably either starting or stopping
                return false;
            }
            return (rt->get_thread_manager().status() >= st);
        }
    }

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
    namespace agas
    {
        // return whether resolver client is in state described by st
        bool router_is(state st)
        {
            runtime_distributed* rt = get_runtime_distributed_ptr();
            if (nullptr == rt) {
                // we're probably either starting or stopping
                return st == state_starting || st == state_stopping;
            }
            return (rt->get_agas_client().get_status() == st);
        }
    }
#endif
}


////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/modules/threadmanager.hpp>
#include <hpx/runtime_local/runtime_local.hpp>
#include <hpx/runtime_local/state.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads {
    // return whether thread manager is in the state described by st
    bool threadmanager_is(state st)
    {
        hpx::runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
        {
            // we're probably either starting or stopping
            return st <= hpx::state::starting || st >= hpx::state::stopping;
        }
        return (rt->get_thread_manager().status() == st);
    }
    bool threadmanager_is_at_least(state st)
    {
        hpx::runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
        {
            // we're probably either starting or stopping
            return false;
        }
        return (rt->get_thread_manager().status() >= st);
    }
}}    // namespace hpx::threads

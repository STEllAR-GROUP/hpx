////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/remote_interface.hpp>

///////////////////////////////////////////////////////////////////////////////
// This must be in global namespace
HPX_REGISTER_PLAIN_ACTION_EX2(
    hpx::agas::server::garbage_collect_non_blocking_remote_action,
    garbage_collect_non_blocking_remote_action, true);

HPX_REGISTER_PLAIN_ACTION_EX2(
    hpx::agas::server::garbage_collect_sync_remote_action,
    garbage_collect_sync_remote_action, true);


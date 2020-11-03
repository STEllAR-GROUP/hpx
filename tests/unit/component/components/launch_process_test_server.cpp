//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>

#include "launch_process_test_server.hpp"

///////////////////////////////////////////////////////////////////////////////
// Test-component which is instantiated by the launching process and registered
// with AGAS such that the launched process can use it.

HPX_REGISTER_ACTION(launch_process_get_message_action);
HPX_REGISTER_ACTION(launch_process_set_message_action);

HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////
// We use a special component registry for this component as it has to be
// disabled by default. All tests requiring this component to be active will
// enable it explicitly.
typedef hpx::components::component<launch_process::test_server> server_type;
HPX_REGISTER_DISABLED_COMPONENT_FACTORY(server_type, launch_process_test_server)
#endif

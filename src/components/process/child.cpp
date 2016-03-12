//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>

#include <hpx/components/process/child.hpp>
#include <hpx/components/process/server/child.hpp>

HPX_REGISTER_ACTION(
    hpx::components::process::server::child::terminate_action,
    hpx_components_process_server_child_terminate_action);

HPX_REGISTER_ACTION(
    hpx::components::process::server::child::wait_for_exit_action,
    hpx_components_process_server_child_wait_for_exit);

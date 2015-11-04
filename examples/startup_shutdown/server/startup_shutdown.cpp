//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include "startup_shutdown.hpp"

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT(
    hpx::components::component<
        startup_shutdown::server::startup_shutdown_component>,
    startup_shutdown_component);

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the simple_accumulator actions
HPX_REGISTER_ACTION(
    startup_shutdown::server::startup_shutdown_component::init_action,
    startup_shutdown_component_init_action);


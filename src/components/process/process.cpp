//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/include/components.hpp>

#include <hpx/components/process/process.hpp>
#include <hpx/components/process/server/child.hpp>

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality, We register the module dynamically
// as no executable links against it.
HPX_REGISTER_COMPONENT_MODULE()

typedef hpx::components::process::server::child child_type;

HPX_REGISTER_COMPONENT(
    hpx::components::component<child_type>,
    hpx_components_process_child_factory, hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(child_type)


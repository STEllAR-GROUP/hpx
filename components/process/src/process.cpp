//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/components_base/server/component.hpp>
#include <hpx/runtime_components/component_factory.hpp>

#include <hpx/components/process/process.hpp>
#include <hpx/components/process/server/child.hpp>

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality, We register the module dynamically
// as no executable links against it.
HPX_REGISTER_COMPONENT_MODULE()

typedef hpx::components::process::server::child child_type;

HPX_REGISTER_COMPONENT(hpx::components::component<child_type>,
    hpx_components_process_child_factory,
    hpx::components::factory_state::enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(child_type)

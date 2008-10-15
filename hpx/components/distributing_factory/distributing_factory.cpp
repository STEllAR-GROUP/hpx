//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/components/distributing_factory/server/distributing_factory.hpp>

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::server::distributing_factory, "distributing_factory");

///////////////////////////////////////////////////////////////////////////////
// For any component derived from manage_component_base we must use the 
// following in exactly one source file
HPX_REGISTER_MANAGED_COMPONENT(hpx::components::server::distributing_factory);


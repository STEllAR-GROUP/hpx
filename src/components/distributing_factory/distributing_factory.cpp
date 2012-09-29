//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>

#include <hpx/components/distributing_factory/server/distributing_factory.hpp>

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE()

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::server::distributing_factory distributing_factory_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<distributing_factory_type>,
    distributing_factory, hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(distributing_factory_type)

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the distributing_factory actions
HPX_REGISTER_ACTION(
    distributing_factory_type::create_components_action,
    distributing_factory_create_components_action)
HPX_REGISTER_ACTION(
    distributing_factory_type::create_partitioned_action,
    distributing_factory_create_partitioned_action)


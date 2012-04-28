//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <hpx/components/distributing_factory/server/distributing_factory.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE()

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::server::distributing_factory distributing_factory_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_EX(
    hpx::components::simple_component<distributing_factory_type>,
    distributing_factory, hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(distributing_factory_type)

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the distributing_factory actions
HPX_REGISTER_ACTION_EX(
    distributing_factory_type::create_components_action,
    distributing_factory_create_components_action)
HPX_REGISTER_ACTION_EX(
    distributing_factory_type::create_partitioned_action,
    distributing_factory_create_partitioned_action)

typedef hpx::lcos::base_lco_with_value<
        distributing_factory_type::remote_result_type
    > create_result_type;

HPX_REGISTER_ACTION_EX(
    create_result_type::set_value_action,
    set_value_action_distributing_factory_result)


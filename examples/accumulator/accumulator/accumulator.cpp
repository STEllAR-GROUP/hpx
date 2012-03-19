//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//[accumulator_cpp
#include <hpx/hpx.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>

#include "server/accumulator.hpp"

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    hpx::components::server::accumulator
> accumulator_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(accumulator_type, accumulator);

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the accumulator actions
HPX_REGISTER_ACTION_EX(
    accumulator_type::wrapped_type::init_action,
    accumulator_init_action);
HPX_REGISTER_ACTION_EX(
    accumulator_type::wrapped_type::add_action,
    accumulator_add_action);
HPX_REGISTER_ACTION_EX(
    accumulator_type::wrapped_type::query_action,
    accumulator_query_action);
HPX_REGISTER_ACTION_EX(
    accumulator_type::wrapped_type::print_action,
    accumulator_print_action);
HPX_DEFINE_GET_COMPONENT_TYPE(accumulator_type::wrapped_type);

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<unsigned long>,
    hpx::components::component_base_lco_with_value);
//]

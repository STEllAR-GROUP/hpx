//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/mpl/print.hpp>

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/managed_accumulator.hpp"

//[managed_accumulator_registration_definitions
///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality.
HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    examples::server::managed_accumulator
> accumulator_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(accumulator_type, managed_accumulator);

///////////////////////////////////////////////////////////////////////////////
// Serialization support for managed_accumulator actions.
HPX_REGISTER_ACTION_EX(
    accumulator_type::wrapped_type::reset_action,
    managed_accumulator_reset_action);
HPX_REGISTER_ACTION_EX(
    accumulator_type::wrapped_type::add_action,
    managed_accumulator_add_action);
HPX_REGISTER_ACTION_EX(
    accumulator_type::wrapped_type::query_action,
    managed_accumulator_query_action);
HPX_DEFINE_GET_COMPONENT_TYPE(accumulator_type::wrapped_type);
//]


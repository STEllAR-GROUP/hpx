//  Copyright (c) 2007-2010 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/simple_accumulator.hpp"

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality.
HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::simple_component<
    examples::server::simple_accumulator
> accumulator_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(accumulator_type, simple_accumulator);

///////////////////////////////////////////////////////////////////////////////
// Serialization support for simple_accumulator actions.
HPX_REGISTER_ACTION_EX(
    accumulator_type::wrapped_type::reset_action,
    simple_accumulator_reset_action);
HPX_REGISTER_ACTION_EX(
    accumulator_type::wrapped_type::add_action,
    simple_accumulator_add_action);
HPX_REGISTER_ACTION_EX(
    accumulator_type::wrapped_type::query_action,
    simple_accumulator_query_action);
HPX_DEFINE_GET_COMPONENT_TYPE(accumulator_type::wrapped_type);


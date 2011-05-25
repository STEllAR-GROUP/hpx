// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>

// Bring in necessary headers for setting up an HPX component
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

// Bring in PXGL headers
#include <pxgl/util/futures.hpp>

#include "pxgl/xua/numeric.hpp"
#include "pxgl/xua/range.hpp"
#include "pxgl/xua/arbitrary_distribution.hpp"

////////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();

////////////////////////////////////////////////////////////////////////////////
// Define item types
typedef double item_type;
typedef std::vector<item_type> items_type;

typedef hpx::naming::id_type id_type;
typedef std::vector<id_type> ids_type;

//typedef int size_type;
typedef unsigned long size_type;

////////////////////////////////////////////////////////////////////////////////
// Define distribution types
typedef pxgl::xua::arbitrary_distribution<
    hpx::naming::id_type,
    pxgl::xua::range
> arbitrary_range_type;

////////////////////////////////////////////////////////////////////////////////
// Define numeric types
typedef pxgl::xua::server::numeric<
    arbitrary_range_type, 
    item_type
> double_numeric_arbitrary_range_type;

////////////////////////////////////////////////////////////////////////////////
// Register component factory for numerics
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::managed_component<double_numeric_arbitrary_range_type>,
    double_numeric_arbitrary_range);

////////////////////////////////////////////////////////////////////////////////
// Add serialization support for block numeric actions
HPX_REGISTER_ACTION_EX(
    double_numeric_arbitrary_range_type::init_action,
    double_numeric_arbitrary_range_init_action);
HPX_REGISTER_ACTION_EX(
    double_numeric_arbitrary_range_type::init_sync_action,
    double_numeric_arbitrary_range_init_sync_action);
HPX_REGISTER_ACTION_EX(
    double_numeric_arbitrary_range_type::ready_action,
    double_numeric_arbitrary_range_ready_action);
HPX_REGISTER_ACTION_EX(
    double_numeric_arbitrary_range_type::ready_all_action,
    double_numeric_arbitrary_range_ready_all_action);
HPX_REGISTER_ACTION_EX(
    double_numeric_arbitrary_range_type::size_action,
    double_numeric_arbitrary_range_size_action);
HPX_REGISTER_ACTION_EX(
    double_numeric_arbitrary_range_type::construct_action,
    double_numeric_arbitrary_range_construct_action);
HPX_REGISTER_ACTION_EX(
    double_numeric_arbitrary_range_type::constructed_action,
    double_numeric_arbitrary_range_constructed_action);
HPX_REGISTER_ACTION_EX(
    double_numeric_arbitrary_range_type::get_distribution_action,
    double_numeric_arbitrary_range_get_distribution_action);
HPX_REGISTER_ACTION_EX(
    double_numeric_arbitrary_range_type::items_action,
    double_numeric_arbitrary_range_items_action);
HPX_REGISTER_ACTION_EX(
    double_numeric_arbitrary_range_type::init_items_action,
    double_numeric_arbitrary_range_init_items_action);
HPX_REGISTER_ACTION_EX(
    double_numeric_arbitrary_range_type::init_incr_action,
    double_numeric_arbitrary_range_init_incr_action);
HPX_REGISTER_ACTION_EX(
    double_numeric_arbitrary_range_type::replicate_action,
    double_numeric_arbitrary_range_replicate_action);
HPX_REGISTER_ACTION_EX(
    double_numeric_arbitrary_range_type::clear_action,
    double_numeric_arbitrary_range_clear_action);
HPX_REGISTER_ACTION_EX(
    double_numeric_arbitrary_range_type::clear_member_action,
    double_numeric_arbitrary_range_clear_member_action);
HPX_REGISTER_ACTION_EX(
    double_numeric_arbitrary_range_type::local_to_action,
    double_numeric_arbitrary_range_local_to_action);

////////////////////////////////////////////////////////////////////////////////
// Define block numeric component
HPX_DEFINE_GET_COMPONENT_TYPE(double_numeric_arbitrary_range_type);

////////////////////////////////////////////////////////////////////////////////
// Add futures support
HPX_REGISTER_FUTURE(item_type, item);
HPX_REGISTER_FUTURE(items_type, items);
HPX_REGISTER_FUTURE(size_type, size);
HPX_REGISTER_FUTURE(arbitrary_range_type, arbitrary_range);
HPX_REGISTER_FUTURE(items_type *, ids_ptr);
HPX_REGISTER_FUTURE(hpx::util::unused_type, unused);


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
#include "../../../pxgl/util/futures.hpp"

#include "../../../pxgl/xua/vector.hpp"
#include "../../../pxgl/xua/range.hpp"
#include "../../../pxgl/xua/constant_distribution.hpp"

// Bring in extra definitions
#include "../../../pxgl/graphs/edge_tuple.hpp"

////////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();

////////////////////////////////////////////////////////////////////////////////
// Define item types
//typedef int size_type;
typedef unsigned long size_type;

typedef pxgl::graphs::server::edge_tuple_type edge_tuple_type;
typedef pxgl::graphs::server::edge_tuples_type edge_tuples_type;

////////////////////////////////////////////////////////////////////////////////
// Define distribution types
typedef pxgl::xua::constant_distribution<
    hpx::naming::id_type,
    pxgl::xua::range
> constant_range_type;

////////////////////////////////////////////////////////////////////////////////
// Define vector types
typedef pxgl::xua::server::vector<
    constant_range_type, 
    edge_tuple_type
> edge_tuple_vector_constant_range_type;

////////////////////////////////////////////////////////////////////////////////
// Register component factory for vectors
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::managed_component<edge_tuple_vector_constant_range_type>,
    edge_tuple_vector_constant_range);

////////////////////////////////////////////////////////////////////////////////
// Add serialization support for block vector actions
HPX_REGISTER_ACTION_EX(
    edge_tuple_vector_constant_range_type::init_action,
    edge_tuple_vector_constant_range_init_action);
HPX_REGISTER_ACTION_EX(
    edge_tuple_vector_constant_range_type::init_sync_action,
    edge_tuple_vector_constant_range_init_sync_action);
HPX_REGISTER_ACTION_EX(
    edge_tuple_vector_constant_range_type::ready_action,
    edge_tuple_vector_constant_range_ready_action);
HPX_REGISTER_ACTION_EX(
    edge_tuple_vector_constant_range_type::ready_all_action,
    edge_tuple_vector_constant_range_ready_all_action);
HPX_REGISTER_ACTION_EX(
    edge_tuple_vector_constant_range_type::size_action,
    edge_tuple_vector_constant_range_size_action);
HPX_REGISTER_ACTION_EX(
    edge_tuple_vector_constant_range_type::construct_action,
    edge_tuple_vector_constant_range_construct_action);
HPX_REGISTER_ACTION_EX(
    edge_tuple_vector_constant_range_type::constructed_action,
    edge_tuple_vector_constant_range_constructed_action);
HPX_REGISTER_ACTION_EX(
    edge_tuple_vector_constant_range_type::get_distribution_action,
    edge_tuple_vector_constant_range_get_distribution_action);
HPX_REGISTER_ACTION_EX(
    edge_tuple_vector_constant_range_type::items_action,
    edge_tuple_vector_constant_range_items_action);
HPX_REGISTER_ACTION_EX(
    edge_tuple_vector_constant_range_type::clear_action,
    edge_tuple_vector_constant_range_clear_action);
HPX_REGISTER_ACTION_EX(
    edge_tuple_vector_constant_range_type::clear_member_action,
    edge_tuple_vector_constant_range_clear_member_action);
HPX_REGISTER_ACTION_EX(
    edge_tuple_vector_constant_range_type::replicate_action,
    edge_tuple_vector_constant_range_replicate_action);
HPX_REGISTER_ACTION_EX(
    edge_tuple_vector_constant_range_type::local_to_action,
    edge_tuple_vector_constant_range_local_to_action);

////////////////////////////////////////////////////////////////////////////////
// Define block vector component
HPX_DEFINE_GET_COMPONENT_TYPE(edge_tuple_vector_constant_range_type);

////////////////////////////////////////////////////////////////////////////////
// Add futures support
HPX_REGISTER_FUTURE(size_type, size);
HPX_REGISTER_FUTURE(constant_range_type, constant_range);
HPX_REGISTER_FUTURE(edge_tuples_type, edge_tuples);
HPX_REGISTER_FUTURE(edge_tuples_type *, edge_tuples_ptr);


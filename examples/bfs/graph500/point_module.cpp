//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/point.hpp"

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    graph500::server::point
> graph500_point_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(graph500_point_type, graph500_point);
HPX_DEFINE_GET_COMPONENT_TYPE(graph500_point_type::wrapped_type);

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION_EX(
    graph500_point_type::wrapped_type::init_action,
    graph500_point_init_action);

HPX_REGISTER_ACTION_EX(
    graph500_point_type::wrapped_type::root_action,
    graph500_point_root_action);

HPX_REGISTER_ACTION_EX(
    graph500_point_type::wrapped_type::ppedge_action,
    graph500_point_ppedge_action);

HPX_REGISTER_ACTION_EX(
    graph500_point_type::wrapped_type::findwhohasthisedge_action,
    graph500_point_findwhohasthisedge_action);

HPX_REGISTER_ACTION_EX(
    graph500_point_type::wrapped_type::receive_duplicates_action,
    graph500_point_receive_duplicates_action);

HPX_REGISTER_ACTION_EX(
    graph500_point_type::wrapped_type::get_parent_action,
    graph500_point_get_parent_action);

HPX_REGISTER_ACTION_EX(
    graph500_point_type::wrapped_type::bfs_action,
    graph500_point_bfs_action);

HPX_REGISTER_ACTION_EX(
    graph500_point_type::wrapped_type::resolve_conflict_action,
    graph500_point_resolve_conflict_action);

HPX_REGISTER_ACTION_EX(
    graph500_point_type::wrapped_type::distributed_validate_action,
    graph500_point_distributed_validate_action);

HPX_REGISTER_ACTION_EX(
    graph500_point_type::wrapped_type::get_numedges_action,
    graph500_point_get_numedges_action);

HPX_REGISTER_ACTION_EX(
    graph500_point_type::wrapped_type::has_edge_action,
    graph500_point_has_edge_action);

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<std::vector<std::size_t> >,
    hpx::components::component_base_lco_with_value);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<std::vector<std::size_t> >::get_value_action,
    get_value_action_vector_size_t);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<std::vector<std::size_t> >::set_result_action,
    set_result_action_vector_size_t);

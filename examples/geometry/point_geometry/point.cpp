//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/actions/continuation.hpp>

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
    hpx::geometry::server::point
> point_geometry_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(point_geometry_type, point_geometry);
HPX_DEFINE_GET_COMPONENT_TYPE(point_geometry_type::wrapped_type);

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the point_geometry actions
HPX_REGISTER_ACTION_EX(
    point_geometry_type::wrapped_type::init_action,
    point_geometry_init_action);
HPX_REGISTER_ACTION_EX(
    point_geometry_type::wrapped_type::move_action,
    point_geometry_move_action);
HPX_REGISTER_ACTION_EX(
    point_geometry_type::wrapped_type::adjust_action,
    point_geometry_adjust_action);
HPX_REGISTER_ACTION_EX(
    point_geometry_type::wrapped_type::enforce_action,
    point_geometry_enforce_action);
HPX_REGISTER_ACTION_EX(
    point_geometry_type::wrapped_type::search_action,
    point_geometry_search_action);
HPX_REGISTER_ACTION_EX(
    point_geometry_type::wrapped_type::recompute_action,
    point_geometry_recompute_action);
HPX_REGISTER_ACTION_EX(
    point_geometry_type::wrapped_type::get_poly_action,
    point_geometry_get_poly_action);
HPX_REGISTER_ACTION_EX(
    point_geometry_type::wrapped_type::get_X_action,
    point_geometry_get_X_action);
HPX_REGISTER_ACTION_EX(
    point_geometry_type::wrapped_type::get_Y_action,
    point_geometry_get_Y_action);
HPX_REGISTER_ACTION_EX(
    point_geometry_type::wrapped_type::set_X_action,
    point_geometry_set_X_action);
HPX_REGISTER_ACTION_EX(
    point_geometry_type::wrapped_type::set_Y_action,
    point_geometry_set_Y_action);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<polygon_type>::set_result_action,
    set_result_action_polygon_type);
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<polygon_type>,
    hpx::components::component_base_lco_with_value);
HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<hpx::geometry::server::vertex_data>::set_result_action,
    set_result_action_vertex_data_type);
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<hpx::geometry::server::vertex_data>,
    hpx::components::component_base_lco_with_value);

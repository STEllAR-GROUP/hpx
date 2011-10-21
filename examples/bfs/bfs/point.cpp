//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>

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
//HPX_REGISTER_ACTION_EX(
//    point_geometry_type::wrapped_type::traverse_action,
//    point_geometry_traverse_action);

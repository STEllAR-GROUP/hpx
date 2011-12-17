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

#include "server/point_tm.hpp"

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
//HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    bfs_tm::server::point
> bfs_tm_point_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(bfs_tm_point_type, bfs_tm_point);
HPX_DEFINE_GET_COMPONENT_TYPE(bfs_tm_point_type::wrapped_type);

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION_EX(
    bfs_tm_point_type::wrapped_type::manager_action,
    bfs_tm_point_manager_action);

HPX_REGISTER_ACTION_EX(
    bfs_tm_point_type::wrapped_type::init_action,
    bfs_tm_point_init_action);

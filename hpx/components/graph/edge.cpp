//  Copyright (c) 2007-2009 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <hpx/components/graph/server/edge.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

///////////////////////////////////////////////////////////////////////////////


typedef hpx::lcos::base_lco_with_value<
        hpx::components::server::edge::edge_snapshot_type
    > create_edge_snapshot_type;

HPX_REGISTER_ACTION_EX(
    create_edge_snapshot_type::set_result_action,
    set_result_action_edge_snapshot_result);
HPX_DEFINE_GET_COMPONENT_TYPE(create_edge_snapshot_type);



///////////////////////////////////////////////////////////////////////////////

typedef hpx::components::managed_component<
    hpx::components::server::edge
> edge_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(edge_type, edge);

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the edge actions
HPX_REGISTER_ACTION_EX(
    edge_type::wrapped_type::init_action,
    edge_init_action);
HPX_REGISTER_ACTION_EX(
    edge_type::wrapped_type::get_snapshot_action,
    edge_get_snapshot_action);
HPX_DEFINE_GET_COMPONENT_TYPE(edge_type::wrapped_type);

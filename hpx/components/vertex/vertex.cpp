//  Copyright (c) 2007-2009 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <hpx/components/vertex/server/vertex.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

//typedef hpx::components::server::vertex::partial_edge_set_type partial_edge_set_type;
typedef hpx::components::server::vertex::partial_edge_set_type partial_edge_set_type;

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    hpx::components::server::vertex
> vertex_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(vertex_type, vertex);

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the vertex actions
HPX_REGISTER_ACTION_EX(
    vertex_type::wrapped_type::init_action,
    vertex_init_action);
HPX_REGISTER_ACTION_EX(
    vertex_type::wrapped_type::label_action,
    vertex_init_action);
HPX_REGISTER_ACTION_EX(
    vertex_type::wrapped_type::add_edge_action,
    vertex_add_edge_action);
HPX_REGISTER_ACTION_EX(
    vertex_type::wrapped_type::out_edges_action,
    vertex_out_edges_action);
HPX_DEFINE_GET_COMPONENT_TYPE(vertex_type::wrapped_type);

//typedef std::vector<std::pair<hpx::naming::id_type, int> > partial_edge_set_type;

typedef hpx::lcos::base_lco_with_value<
        hpx::components::server::vertex::partial_edge_set_type
    > create_partial_edge_set_type;

HPX_REGISTER_ACTION_EX(
    create_partial_edge_set_type::set_result_action,
    set_result_action_vertex_result);
HPX_DEFINE_GET_COMPONENT_TYPE(create_partial_edge_set_type);

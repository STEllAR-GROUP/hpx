//  Copyright (c) 2007-2009 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <hpx/components/graph/server/graph.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>


///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    hpx::components::server::graph
> graph_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(graph_type, graph);

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the graph actions
HPX_REGISTER_ACTION_EX(
    graph_type::wrapped_type::init_action,
    graph_init_action);
HPX_REGISTER_ACTION_EX(
    graph_type::wrapped_type::order_action,
    graph_order_action);
HPX_REGISTER_ACTION_EX(
    graph_type::wrapped_type::size_action,
    graph_size_action);
HPX_REGISTER_ACTION_EX(
	graph_type::wrapped_type::add_edge_action,
	graph_add_edge_action);
HPX_REGISTER_ACTION_EX(
	graph_type::wrapped_type::vertex_name_action,
	graph_vertex_name_action);
HPX_REGISTER_ACTION_EX(
    graph_type::wrapped_type::vertices_action,
    graph_vertices_action);
HPX_DEFINE_GET_COMPONENT_TYPE(graph_type::wrapped_type);

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

// Bring in graph and aux. definition
#include <pxgl/util/futures.hpp>
#include "pxgl/graphs/csr_graph.hpp"
#include "pxgl/xua/range.hpp"
#include "pxgl/xua/arbitrary_distribution.hpp"
#include "pxgl/xua/vector.hpp"

#include "pxgl/graphs/edge_tuple.hpp"

typedef unsigned long size_type;
typedef std::vector<size_type> sizes_type;

typedef hpx::naming::gid_type gid_type;

typedef pxgl::graphs::server::edge_tuple_type edge_tuple_type;
typedef pxgl::graphs::server::edge_tuples_type edge_tuples_type;

////////////////////////////////////////////////////////////////////////////////
// Define distribution types
typedef pxgl::xua::arbitrary_distribution<
    hpx::naming::id_type,
    pxgl::xua::range
> arbitrary_range_type;

////////////////////////////////////////////////////////////////////////////////
// Define vector types
typedef pxgl::xua::vector<
    arbitrary_range_type,
    edge_tuple_type
> edge_tuple_vector_arbitrary_range_client_type;

typedef hpx::components::managed_component<
  pxgl::graphs::server::csr_graph<edge_tuple_vector_arbitrary_range_client_type, arbitrary_range_type>
> csr_graph_arb_arb_type;

typedef pxgl::graphs::server::signal_value_type signal_value_type;
typedef pxgl::graphs::server::edge_iterator<
    edge_tuples_type, 
    size_type
>
edge_iterator_type;
typedef std::vector<pxgl::graphs::server::adjacency<size_type, double> > 
    adjacencies_type;

////////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();

////////////////////////////////////////////////////////////////////////////////
// Register component factory for CSR graphs
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    csr_graph_arb_arb_type, 
    csr_graph_arb_arb);

////////////////////////////////////////////////////////////////////////////////
// Add serialization support for CSR graph actions
HPX_REGISTER_ACTION_EX(
    csr_graph_arb_arb_type::wrapped_type::finalize_init_action,
    csr_graph_arb_arb_finalize_init_action);
HPX_REGISTER_ACTION_EX(
    csr_graph_arb_arb_type::wrapped_type::aligned_init_action,
    csr_graph_arb_arb_aligned_init_action);
HPX_REGISTER_ACTION_EX(
    csr_graph_arb_arb_type::wrapped_type::signal_init_action,
    csr_graph_arb_arb_signal_init_action);
HPX_REGISTER_ACTION_EX(
    csr_graph_arb_arb_type::wrapped_type::init_local_action,
    csr_graph_arb_arb_init_local_action);
HPX_REGISTER_ACTION_EX(
    csr_graph_arb_arb_type::wrapped_type::init_action,
    csr_graph_arb_arb_init_action);
HPX_REGISTER_ACTION_EX(
    csr_graph_arb_arb_type::wrapped_type::add_local_vertices_action,
    csr_graph_arb_arb_add_local_vertices_action);
HPX_REGISTER_ACTION_EX(
    csr_graph_arb_arb_type::wrapped_type::ready_action,
    csr_graph_arb_arb_ready_action);
HPX_REGISTER_ACTION_EX(
    csr_graph_arb_arb_type::wrapped_type::ready_all_action,
    csr_graph_arb_arb_ready_all_action);
HPX_REGISTER_ACTION_EX(
    csr_graph_arb_arb_type::wrapped_type::order_action,
    csr_graph_arb_arb_order_action);
HPX_REGISTER_ACTION_EX(
    csr_graph_arb_arb_type::wrapped_type::size_action,
    csr_graph_arb_arb_size_action);
HPX_REGISTER_ACTION_EX(
    csr_graph_arb_arb_type::wrapped_type::vertices_action,
    csr_graph_arb_arb_vertices_action);
HPX_REGISTER_ACTION_EX(
    csr_graph_arb_arb_type::wrapped_type::edges_action,
    csr_graph_arb_arb_edges_action);
HPX_REGISTER_ACTION_EX(
    csr_graph_arb_arb_type::wrapped_type::construct_action,
    csr_graph_arb_arb_construct_action);
HPX_REGISTER_ACTION_EX(
    csr_graph_arb_arb_type::wrapped_type::get_distribution_action,
    csr_graph_arb_arb_get_distribution_action);
HPX_REGISTER_ACTION_EX(
    csr_graph_arb_arb_type::wrapped_type::replicate_action,
    csr_graph_arb_arb_replicate_action);
HPX_REGISTER_ACTION_EX(
    csr_graph_arb_arb_type::wrapped_type::local_to_action,
    csr_graph_arb_arb_local_to_action);
HPX_REGISTER_ACTION_EX(
    csr_graph_arb_arb_type::wrapped_type::neighbors_action,
    csr_graph_arb_arb_neighbors_action);

////////////////////////////////////////////////////////////////////////////////
// Define CSR graph component
HPX_DEFINE_GET_COMPONENT_TYPE(csr_graph_arb_arb_type::wrapped_type);

////////////////////////////////////////////////////////////////////////////////
// Add futures support
HPX_REGISTER_FUTURE(size_type, size);
HPX_REGISTER_FUTURE(edge_tuples_type, edge_tuples);
HPX_REGISTER_FUTURE(edge_tuples_type *, edges_ptr);
HPX_REGISTER_FUTURE(arbitrary_range_type, arbitrary_range);
HPX_REGISTER_FUTURE(signal_value_type, signal_value);
HPX_REGISTER_FUTURE(edge_iterator_type, edge_iterator);
HPX_REGISTER_FUTURE(adjacencies_type *, adjacencies_ptr);
HPX_REGISTER_FUTURE(sizes_type *, sizes_ptr);


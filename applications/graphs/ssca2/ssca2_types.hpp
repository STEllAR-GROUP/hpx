//  Copyright (c) 2009-2010 Dylan Stark
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLICATIONS_SSCA2_TYPES_SEP_28_2009_0353PM)
#define HPX_APPLICATIONS_SSCA2_TYPES_SEP_28_2009_0353PM

#include <hpx/hpx.hpp>

#include <hpx/components/distributing_factory/distributing_factory.hpp>

#include <hpx/components/graph/graph.hpp>
#include <hpx/components/graph/edge.hpp>
#include <hpx/components/vertex/vertex.hpp>

#include <hpx/components/distributed_set/distributed_set.hpp>
#include <hpx/components/distributed_set/local_set.hpp>
#include <hpx/components/distributed_set/server/distributed_set.hpp>
#include <hpx/components/distributed_set/server/local_set.hpp>

#include <hpx/components/distributed_map/distributed_map.hpp>
#include <hpx/components/distributed_map/local_map.hpp>

#include "props/props.hpp"
#include "reduce_max/reduce_max.hpp"

///////////////////////////////////////////////////////////////////////////////
// HPX types

typedef hpx::naming::id_type gid_type;
typedef std::vector<gid_type> gids_type;
typedef std::map<gid_type,gid_type> gids_map_type;

typedef hpx::lcos::future_value<int> future_int_type;
typedef std::vector<future_int_type> future_ints_type;

typedef hpx::lcos::future_value<gid_type> future_gid_type;
typedef std::vector<future_gid_type> future_gids_type;

///////////////////////////////////////////////////////////////////////////////
// Misc. component types

typedef hpx::components::server::distributing_factory distributing_factory_type;

///////////////////////////////////////////////////////////////////////////////
// Graph-related types

typedef hpx::components::server::graph graph_type;
typedef hpx::components::stubs::graph stub_graph_type;
typedef hpx::components::graph client_graph_type;

typedef hpx::components::server::distributed_set<graph_type> dist_graph_set_type;
typedef hpx::components::stubs::distributed_set<graph_type> stub_dist_graph_set_type;
typedef hpx::components::distributed_set<graph_type> client_dist_graph_set_type;
typedef hpx::components::server::local_set<graph_type> local_graph_set_type;
typedef hpx::components::stubs::local_set<graph_type>  stub_local_graph_set_type;
typedef hpx::components::local_set<graph_type>         client_local_graph_set_type;

///////////////////////////////////////////////////////////////////////////////
// Vertex-related types

typedef hpx::components::server::vertex vertex_type;
typedef hpx::components::stubs::vertex stub_vertex_type;
typedef hpx::components::server::distributed_set<vertex_type> dist_vertex_set_type;
typedef hpx::components::stubs::distributed_set<vertex_type> stub_dist_vertex_set_type;
typedef hpx::components::distributed_set<vertex_type> client_dist_vertex_set_type;
typedef hpx::components::server::local_set<vertex_type> local_vertex_set_type;
typedef hpx::components::stubs::local_set<vertex_type> stub_local_vertex_set_type;

///////////////////////////////////////////////////////////////////////////////
// Edge-related types

typedef hpx::components::server::edge edge_type;
typedef hpx::components::stubs::edge stub_edge_type;

typedef hpx::components::server::distributed_set<edge_type> dist_edge_set_type;
typedef hpx::components::server::local_set<edge_type> local_edge_set_type;
typedef hpx::components::stubs::local_set<edge_type> stub_local_edge_set_type;
typedef hpx::components::distributed_set<edge_type> client_dist_edge_set_type;

///////////////////////////////////////////////////////////////////////////////
// Map-related types

typedef hpx::components::server::distributed_map<gids_map_type> dist_gids_map_type;
typedef hpx::components::server::local_map<gids_map_type> local_gids_map_type;
typedef hpx::components::stubs::distributed_map<gids_map_type> stub_dist_gids_map_type;
typedef hpx::components::distributed_map<gids_map_type> client_dist_gids_map_type;
typedef hpx::components::stubs::local_map<gids_map_type> stub_local_gids_map_type;

///////////////////////////////////////////////////////////////////////////////
// Props component types

typedef hpx::components::props client_props_type;
typedef hpx::components::server::props props_type;

#endif

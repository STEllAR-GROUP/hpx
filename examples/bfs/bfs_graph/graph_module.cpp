//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/components.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/graph.hpp"
#include "server/bgl_graph.hpp"
#include "server/concurrent_bgl_graph.hpp"

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    bfs::server::graph
> bfs_graph_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(bfs_graph_type, bfs_graph);

HPX_REGISTER_ACTION(
    bfs_graph_type::wrapped_type::init_action,
    bfs_graph_init_action);
HPX_REGISTER_ACTION(
    bfs_graph_type::wrapped_type::bfs_action,
    bfs_graph_bfs_action);
HPX_REGISTER_ACTION(
    bfs_graph_type::wrapped_type::get_parents_action,
    bfs_graph_get_parents_action);
HPX_REGISTER_ACTION(
    bfs_graph_type::wrapped_type::reset_action,
    bfs_graph_reset_action);

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    bfs::server::bgl_graph
> bfs_bgl_graph_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(bfs_bgl_graph_type, bfs_bgl_graph);

HPX_REGISTER_ACTION(
    bfs_bgl_graph_type::wrapped_type::init_action,
    bfs_bgl_graph_init_action);
HPX_REGISTER_ACTION(
    bfs_bgl_graph_type::wrapped_type::bfs_action,
    bfs_bgl_graph_bfs_action);
HPX_REGISTER_ACTION(
    bfs_bgl_graph_type::wrapped_type::get_parents_action,
    bfs_bgl_graph_get_parents_action);
HPX_REGISTER_ACTION(
    bfs_bgl_graph_type::wrapped_type::reset_action,
    bfs_bgl_graph_reset_action);

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    bfs::server::concurrent_bgl_graph
> bfs_concurrent_bgl_graph_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    bfs_concurrent_bgl_graph_type, bfs_concurrent_bgl_graph);

HPX_REGISTER_ACTION(
    bfs_concurrent_bgl_graph_type::wrapped_type::init_action,
    bfs_concurrent_bgl_graph_init_action);
HPX_REGISTER_ACTION(
    bfs_concurrent_bgl_graph_type::wrapped_type::bfs_action,
    bfs_concurrent_bgl_graph_bfs_action);
HPX_REGISTER_ACTION(
    bfs_concurrent_bgl_graph_type::wrapped_type::get_parents_action,
    bfs_concurrent_bgl_graph_get_parents_action);
HPX_REGISTER_ACTION(
    bfs_concurrent_bgl_graph_type::wrapped_type::reset_action,
    bfs_concurrent_bgl_graph_reset_action);


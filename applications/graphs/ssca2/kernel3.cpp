//  Copyright (c) 2009-2010 Dylan Stark
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <queue>

#include <stdlib.h>
#include <time.h>

#include <hpx/hpx.hpp>
#include <hpx/util/logging.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

#include <hpx/lcos/future_wait.hpp>
#include <hpx/lcos/eager_future.hpp>

#include <hpx/components/graph/graph.hpp>
#include <hpx/components/graph/edge.hpp>
#include <hpx/components/vertex/vertex.hpp>

#include <hpx/components/distributed_set/distributed_set.hpp>
#include <hpx/components/distributed_set/local_set.hpp>
#include <hpx/components/distributed_set/server/distributed_set.hpp>
#include <hpx/components/distributed_set/server/local_set.hpp>

#include <hpx/components/distributed_map/distributed_map.hpp>
#include <hpx/components/distributed_map/local_map.hpp>

#include "ssca2_benchmark.hpp"

#include "props/props.hpp"

#include <boost/unordered_map.hpp>

using namespace hpx;

#define LSSCA_(lvl) LAPP_(lvl) << " [SSCA] "

///////////////////////////////////////////////////////////////////////////////

HPX_REGISTER_ACTION(kernel3_action);

int kernel3(naming::id_type edge_set, naming::id_type subgraphs)
{
    LSSCA_(info) << "event: action(ssca2::extract) status(begin)";
    LSSCA_(info) << "parent(" << threads::get_parent_id() << ")";

    LSSCA_(info) << "extract(" << edge_set
                 << ", " << subgraphs << ")";

    std::vector<lcos::future_value<int> > results;

    // Get vector of local sublists of the edge set
    std::vector<naming::id_type> locals =
        lcos::eager_future<
            dist_edge_set_type::locals_action
        >(edge_set).get();

    // Spawn extract_local local to each sublist of the edge set
    std::vector<naming::id_type>::const_iterator end = locals.end();
    for (std::vector<naming::id_type>::const_iterator it = locals.begin();
         it != end; ++it)
    {
        // This uses hack to get prefix
        naming::id_type locale(boost::uint64_t((*it).get_msb()) << 32, 0);

        // Get colocated portion of subgraphs set
        naming::id_type local_subgraphs =
            lcos::eager_future<
                dist_graph_set_type::get_local_action
            >(subgraphs, locale).get();

        // Spawn actions local to data
        results.push_back(lcos::eager_future<
            extract_local_action
        >(locale, *it, local_subgraphs));
    }

    // Collect notifications that local actions have finished
    std::vector<lcos::future_value<int> >::iterator rend = results.end();
    for (std::vector<lcos::future_value<int> >::iterator rit = results.begin();
         rit != rend; ++rit)
    {
        (*rit).get();
    }

    return 0;
}

HPX_REGISTER_ACTION(extract_local_action);

int extract_local(naming::id_type local_edge_set,
                     naming::id_type local_subgraphs)
{
    LSSCA_(info) << "event: action(ssca2::extract_local) status(begin)";
    LSSCA_(info) << "parent(" << threads::get_parent_id() << ")";

    LSSCA_(info) << "extract_local(" << local_edge_set
                 << ", " << local_subgraphs << ")";

    // Get local vector of edges, for iterating over
    std::vector<naming::id_type> edges(
        lcos::eager_future<
            local_edge_set_type::get_action
        >(local_edge_set).get());

    // Allocate vector of empty subgraphs
    // This is creating the actual individual subgraphs
    // (mirroring the local portion of the edge list)
    //std::vector<naming::id_type> graph_set_local(edges.size());
    // This uses hack to get prefix

    // Why are we using a dist-set for subgraphs?
    // Why not a pmap?
    // That would give a mapping from e->S_e.

    // Should be not be syncing until as late as possible,
    // after the pmaps returns
    naming::id_type here(boost::uint64_t(local_edge_set.get_msb()) << 32,0);
    std::vector<naming::id_type> graphs;
    for (int i=0; i < edges.size(); ++i)
    {
        graphs.push_back(lcos::eager_future<
            local_graph_set_type::add_item_action
        >(local_subgraphs, naming::invalid_id).get());
    }

    // Allocate vector of property maps for each subgraph
    // This is creating the actual individual property maps
    std::vector<naming::id_type> pmaps;

    for (int i = 0; i < edges.size(); ++i)
    {
        pmaps.push_back(stub_dist_gids_map_type::create(here));
    }

    // Append local vector of graphs
    // Think about "grow" action to expand with N new items
    // (Could replace this with local_set<>::add_item()'s)
    /*
    lcos::eager_future<
        local_graph_set_type::append_action
    >(local_subgraphs, graph_set_local).get();
    */

    // Extract subgraphs for each edge
    std::vector<naming::id_type> Hs;
    std::vector<naming::id_type> srcs;
    std::vector<lcos::future_value<naming::id_type> > results;

    int i = 0;
    std::vector<naming::id_type>::const_iterator end = edges.end();
    for (std::vector<naming::id_type>::const_iterator it = edges.begin();
         it != end; ++it, ++i)
    {
        // This is per edge/subgraph/pmap

        Hs.push_back(graphs[i]);
        int d = 3; // This should be an argument

        edge_type::edge_snapshot_type e(
            lcos::eager_future<
                components::server::edge::get_snapshot_action
            >(*it).get());

        // We are local to source

        // Get pmap local to source
        // This uses hack to get prefix
        naming::id_type locale(boost::uint64_t(e.source_.get_msb()) << 32,0);
        naming::id_type local_pmap =
            stub_dist_gids_map_type::get_local(pmaps[i], locale);

        LSSCA_(info) << "Got local pmap " << local_pmap;

        // Get source from local_pmap
        components::component_type props_comp_type =
            components::get_component_type<props_type>();
        naming::id_type source_props =
            stub_local_gids_map_type::value(local_pmap, e.source_, props_comp_type);

        LSSCA_(info) << "Got source_props " << source_props;

        // Add source to H
        srcs.push_back(lcos::eager_future<
            graph_type::add_vertex_action
        >(Hs.back(), naming::invalid_id).get());

        // Get the color of the source vertex
        int color =
            lcos::eager_future<
                props_type::color_action
            >(source_props, d).get();


        if (color >= d && d > 1)
        {
            // Spawn subgraph extraction local to target
            // Probably should rework this to use continuations :-)
            naming::id_type there(boost::uint64_t(e.target_.get_msb()) << 32,0);
            results.push_back(
                lcos::eager_future<
                    extract_subgraph_action
                >(there, Hs.back(), pmaps[i], e.source_, e.target_, d)
            );
        }
    }

    std::vector<lcos::future_value<int> > more_results;

    // Collect notifications that subgraph extractions have finished
    while (Hs.size() > 0)
    {
        more_results.push_back(lcos::eager_future<
            graph_type::add_edge_action
        >(Hs.back(), srcs.back(), results.back().get(), -1)); // We lost the label

        Hs.pop_back();
        srcs.pop_back();
        results.pop_back();
    }

    // Can't let the add_edge go out of scope before the result makes
    // it to the future_value
    while (more_results.size() > 0)
    {
        more_results.back().get();
        more_results.pop_back();
    }

         /*
    std::vector<lcos::future_value<naming::id_type> >::iterator rend = results.end();
    for (std::vector<lcos::future_value<naming::id_type> >::iterator rit = results.begin();
         rit != rend; ++rit)
    {
        naming::id_type new_target = (*rit).get();

        // Can do this because we know the vertices were already added


    }
    */

    return 0;
}

HPX_REGISTER_ACTION(extract_subgraph_action);

naming::id_type extract_subgraph(naming::id_type H, naming::id_type pmap,
                        naming::id_type source, naming::id_type target,
                        int d)
{
    LSSCA_(info) << "event: action(ssca2::extract_subgraph ) status(begin)";
    LSSCA_(info) << "parent(" << threads::get_parent_id() << ")";

    LSSCA_(info) << "extract_subgraph(" << H << ", " << pmap
                 << ", " << source << ", " << target << ", " << d << ")";

    // Note: execution is local to target

    // Get pmap local to target
    // This uses hack to get prefix
    naming::id_type locale(boost::uint64_t(target.get_msb()) << 32,0);
    naming::id_type local_pmap =
        stub_dist_gids_map_type::get_local(pmap, locale);

    LSSCA_(info) << "Got local pmap " << local_pmap;

    // Get target from local_pmap
    components::component_type props_comp_type =
        components::get_component_type<props_type>();
    naming::id_type target_props =
        stub_local_gids_map_type::value(local_pmap, target, props_comp_type);

    LSSCA_(info) << "Got target_props " << target_props;

    // Add (new) source (i.e., the old target) to H
    naming::id_type new_source = lcos::eager_future<
        graph_type::add_vertex_action
    >(H, naming::invalid_id).get();

    // Get the color of the source vertex
    int color =
        lcos::eager_future<
            props_type::color_action
        >(target_props, d).get();

    if (color >= d && d > 1)
    {
        // Continue with the search
        std::vector<lcos::future_value<naming::id_type> > results;

        partial_edge_set_type neighbors =
            lcos::eager_future<vertex_type::out_edges_action>(target).get();

        LSSCA_(info) << "Visiting " << neighbors.size()
                     << " neighbors of target " << target;

        partial_edge_set_type::iterator end = neighbors.end();
        for (partial_edge_set_type::iterator it = neighbors.begin();
             it != end; ++it)
        {
            // Spawn subsequent search
            results.push_back(
                lcos::eager_future<
                    extract_subgraph_action
                >(locale, H, pmap, target, (*it).target_, d-1)
            );
        }

        std::vector<lcos::future_value<int> > add_edge_fs;

        // Collect notifications of when subsequent searches are finished
        std::vector<lcos::future_value<naming::id_type> >::iterator rend = results.end();
        for (std::vector<lcos::future_value<naming::id_type> >::iterator rit = results.begin();
             rit != rend; ++rit)
        {
            naming::id_type new_target = (*rit).get();

            // Can do this because we know the vertices were already added
            add_edge_fs.push_back(lcos::eager_future<
                    graph_type::add_edge_action
                >(H, new_source, new_target, -1)); // We lost the label
        }
        while (add_edge_fs.size() > 0)
        {
            add_edge_fs.back().get();
            add_edge_fs.pop_back();
        }
    }

    return new_source;
}

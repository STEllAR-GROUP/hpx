//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/cstdint.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

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

#include <hpx/lcos/eager_future.hpp>
#include "../../reduce_max/reduce_max.hpp"
#include "../../pbreak/pbreak.hpp"

#include "ssca2.hpp"
#include "../stubs/ssca2.hpp"

#include "../../props/props.hpp"

///////////////////////////////////////////////////////////////////////////////

typedef hpx::components::server::ssca2 ssca2_type;

typedef hpx::components::edge edge_type;
typedef hpx::components::server::distributed_set<edge_type> dist_edge_set_type;
typedef hpx::components::server::local_set<edge_type> local_edge_set_type;

typedef hpx::components::graph graph_type;
typedef hpx::components::server::distributed_set<graph_type> dist_graph_set_type;
typedef hpx::components::server::local_set<graph_type> local_graph_set_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the ssca2 actions
HPX_REGISTER_ACTION_EX(ssca2_type::large_set_action,
                       ssca2_large_set_action);
HPX_REGISTER_ACTION_EX(ssca2_type::large_set_local_action,
                       ssca2_large_set_local_action);
HPX_REGISTER_ACTION_EX(ssca2_type::extract_action,
                       ssca2_extract_action);
HPX_REGISTER_ACTION_EX(ssca2_type::extract_local_action,
                       ssca2_extract_local_action);
HPX_REGISTER_ACTION_EX(ssca2_type::extract_subgraph_action,
                       ssca2_extract_subgraph_action);
HPX_REGISTER_ACTION_EX(ssca2_type::init_props_map_action,
                       ssca2_init_props_map_action);
HPX_REGISTER_ACTION_EX(ssca2_type::init_props_map_local_action,
                       ssca2_init_props_map_local_action);
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<ssca2_type>, ssca2);
HPX_DEFINE_GET_COMPONENT_TYPE(ssca2_type);

/*
typedef hpx::lcos::base_lco_with_value<
        ssca2_type::edge_set_type
    > create_edge_set_type;

HPX_REGISTER_ACTION_EX(
    create_edge_set_type::set_result_action,
    set_result_action_ssca2_result);
HPX_DEFINE_GET_COMPONENT_TYPE(create_edge_set_type);
*/

#define LSSCA_(lvl) LAPP_(lvl) << " [SSCA] "

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ssca2::ssca2() {}
    
    int
    ssca2::large_set(naming::id_type G, naming::id_type dist_edge_set)
    {
        LSSCA_(info) << "large_set(" << G << ", " << dist_edge_set << ")";

        typedef components::distributing_factory::result_type result_type;

        int total_added = 0;
        std::vector<lcos::future_value<int> > local_searches;

        // Get vertex set of G
        result_type vertices =
            lcos::eager_future<graph::vertices_action>(G).get();

        // Create lcos to reduce max value
        lcos::reduce_max local_max(vertices.size(), 1);
        lcos::reduce_max global_max(1, vertices.size());

        // Spawn searches on local sub lists
        result_type::const_iterator end = vertices.end();
        for (result_type::const_iterator it = vertices.begin();
             it != end; ++it)
        {
            local_searches.push_back(
                lcos::eager_future<large_set_local_action>((*it).prefix_,
                    *it,
                    dist_edge_set,
                    local_max.get_gid(),
                    global_max.get_gid()));
        }

        // Get the (reduced) global maximum, and signal it back to
        // local searches
        typedef hpx::lcos::detail::reduce_max::wait_action wait_action;
        global_max.signal(
            lcos::eager_future<wait_action>(local_max.get_gid()).get());

        // Run through local search results to tally total edges added
        while (local_searches.size() > 0)
        {
            total_added += local_searches.back().get();
            local_searches.pop_back();
        }

        LSSCA_(info) << "Found large set of size " << total_added;

        return total_added;
    }
    
    int
    ssca2::large_set_local(locality_result local_vertex_set,
                             naming::id_type edge_set,
                             naming::id_type local_max_lco,
                             naming::id_type global_max_lco)
    {
        LSSCA_(info) << "large_set_local(local_vertex_set, " << edge_set
                     << ", " << local_max_lco << ", " << global_max_lco << ")";

        int max = -1;
        int num_added = 0;

        std::vector<edge_type::edge_snapshot_type> edge_set_local;

        // Iterate over local edges
        naming::id_type gid = local_vertex_set.first_gid_;
        for (std::size_t cnt = 0; cnt < local_vertex_set.count_; ++cnt, ++gid)
        {
            // Get incident edges from this vertex
            typedef vertex::partial_edge_set_type partial_type;
            partial_type partials =
                lcos::eager_future<vertex::out_edges_action>(gid).get();

            // Iterate over incident edges
            partial_type::iterator end = partials.end();
            for (partial_type::iterator it = partials.begin(); it != end; ++it)
            {
                if ((*it).label_ > max)
                {
                    edge_set_local.clear();

                    edge_type::edge_snapshot_type e(gid, (*it).target_, (*it).label_);

                    edge_set_local.push_back(e);
                    //edge_set_local.push_back(
                    //    edge(gid, (*it).target_, (*it).label_));
                    max = (*it).label_;
                }
                else if ((*it).label_ == max)
                {
                    edge_set_local.push_back(
                        edge::edge_snapshot_type(gid, (*it).target_, (*it).label_));
                }
            }
        }

        LSSCA_(info) << "Max on locality " << local_vertex_set.prefix_ << " "
                    << "is " << max << ", total of " << edge_set_local.size();

        // Signal local maximum
        applier::apply<
            lcos::detail::reduce_max::signal_action
        >(local_max_lco, max);

        LSSCA_(info) << "Waiting to see max is on locality "
                    << local_vertex_set.prefix_;

        // Wait for global maximum
        int global_max =
            lcos::eager_future<
                hpx::lcos::detail::reduce_max::wait_action
            >(global_max_lco).get();

        // Add local max edge set if it is globally maximal
        if (max == global_max)
        {
            LSSCA_(info) << "Adding local edge set at "
                        << local_vertex_set.prefix_;

            /*
            typedef distributed_set<ssca2::edge_type>
                distributed_edge_set_type;
            typedef local_set<ssca2::edge_set_type> local_edge_set_type;
            */

            naming::id_type local_set =
                lcos::eager_future<
                    dist_edge_set_type::get_local_action
                >(edge_set, local_vertex_set.prefix_).get();

            // Convert edge snapshots into real edges

            naming::id_type edge_base(stubs::edge::create(local_vertex_set.prefix_, edge_set_local.size()));
            std::vector<naming::id_type> edges(edge_set_local.size());

            int i=0;
            std::vector<edge_type::edge_snapshot_type>::const_iterator eend = edge_set_local.end();
            for (std::vector<edge_type::edge_snapshot_type>::const_iterator eit = edge_set_local.begin();
                 eit != eend; ++eit, ++i)
            {
                lcos::eager_future<
                               server::edge::init_action
                           >(edge_base+i, (*eit).source_, (*eit).target_, (*eit).label_).get();
                edges[i] = edge_base+i;
            }

            num_added =
                lcos::eager_future<
                    local_edge_set_type::append_action
                >(local_set, edges).get();
        }
        else
        {
            LSSCA_(info) << "Not adding local edge set at "
                        << local_vertex_set.prefix_;
        }

        return num_added;
    }

    int
    ssca2::extract(naming::id_type edge_set, naming::id_type subgraphs)
    {
        LSSCA_(info) << "extract(" << edge_set
                     << ", " << subgraphs << ")";

        /*
        typedef components::edge edge;
        typedef components::graph graph;

        typedef components::server::distributed_set distributed_set;
        */

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

    int
    ssca2::extract_local(naming::id_type local_edge_set,
                         naming::id_type local_subgraphs)
    {
        LSSCA_(info) << "extract_local(" << local_edge_set
                     << ", " << local_subgraphs;

        /*
        typedef components::edge edge;
        typedef components::graph graph;
        typedef components::server::local_set local_set;
        */

        // Get local vector of edges, for iterating over
        std::vector<naming::id_type> edges(
            lcos::eager_future<
                local_edge_set_type::get_action
            >(local_edge_set).get());

        // Allocate vector of empty subgraphs
        // (mirroring the local portion of the edge list)
        std::vector<naming::id_type> graph_set_local(edges.size());

        // This uses hack to get prefix
        naming::id_type here(boost::uint64_t(local_edge_set.get_msb()) << 32,0);
        naming::id_type graphs = components::stubs::graph::create(here, edges.size());

        // Allocate vector of property maps for each subgraph
        typedef std::map<naming::id_type,naming::id_type> gids_map_type;
        typedef stubs::distributed_map<gids_map_type> dist_gids_map_type;
        std::vector<naming::id_type> pmaps;

        for (int i = 0; i < edges.size(); ++i)
        {
            graph_set_local.push_back(graphs + i);
            pmaps.push_back(dist_gids_map_type::create(here));
        }

        // Append local vector of graphs
        // Think about "grow" action to expand with N new items
        lcos::eager_future<
            local_graph_set_type::append_action
        >(local_subgraphs, graph_set_local).get();

        // Extract subgraphs for each edge
        std::vector<lcos::future_value<int> > results;

        int i = 0;
        std::vector<naming::id_type>::const_iterator end = edges.end();
        for (std::vector<naming::id_type>::const_iterator it = edges.begin();
             it != end; ++it, ++i)
        {
            naming::id_type H = graphs + i;
            int d = 3; // This should be an argument

            edge_type::edge_snapshot_type e(
                lcos::eager_future<
                    components::server::edge::get_snapshot_action
                >(*it).get());

            // We are local to source

            // Get pmap local to source
            // This is a consequence of not really being able to continue
            // the action local to the data

            // This uses hack to get prefix
            naming::id_type locale(boost::uint64_t(e.source_.get_msb()) << 32,0);
            naming::id_type local_pmap =
                dist_gids_map_type::get_local(pmaps[i], locale);

            LSSCA_(info) << "Got local pmap " << local_pmap;

            // Get source from local_pmap
            components::component_type props_type =
                components::get_component_type<components::server::props>();
            naming::id_type source_props =
                components::stubs::local_map<gids_map_type>::value(local_pmap, e.source_, props_type);

            LSSCA_(info) << "Got source_props " << source_props;

            // Get the color of the source vertex
            int color =
                lcos::eager_future<
                    server::props::color_action
                >(source_props, d).get();


            if (color >= d && d > 1)
            {
                // Spawn subgraph extraction local to target
                naming::id_type there(boost::uint64_t(e.target_.get_msb()) << 32,0);
                results.push_back(lcos::eager_future<
                    extract_subgraph_action
                >(there, H, pmaps[i], e.source_, e.target_, d));
            }
        }

        // Collect notifications that subgraph extractions have finished
        std::vector<lcos::future_value<int> >::iterator rend = results.end();
        for (std::vector<lcos::future_value<int> >::iterator rit = results.begin();
             rit != rend; ++rit)
        {
            (*rit).get();
        }

        return 0;
    }

    int
    ssca2::extract_subgraph(naming::id_type H, naming::id_type pmap,
                            naming::id_type source, naming::id_type target,
                            int d)
    {
        typedef std::map<naming::id_type,naming::id_type> gids_map_type;
        typedef components::stubs::local_map<gids_map_type> local_gids_map_type;
        typedef stubs::distributed_map<gids_map_type> dist_gids_map_type;

        LSSCA_(info) << "extract_subgraph(" << H << ", " << pmap
                     << ", " << source << ", " << target << ", " << d << ")";

        // Note: execution is local to target

        // Get pmap local to target

        // This uses hack to get prefix
        naming::id_type locale(boost::uint64_t(target.get_msb()) << 32,0);
        naming::id_type local_pmap =
            dist_gids_map_type::get_local(pmap, locale);

        LSSCA_(info) << "Got local pmap " << local_pmap;

        // Get target from local_pmap
        components::component_type props_type =
            components::get_component_type<components::server::props>();
        naming::id_type target_props =
            local_gids_map_type::value(local_pmap, target, props_type);

        LSSCA_(info) << "Got target_props " << target_props;

        // Get the color of the source vertex
        int color =
            lcos::eager_future<
                server::props::color_action
            >(target_props, d).get();

        if (color >= d && d > 1)
        {
            // Continue with the search
            std::vector<lcos::future_value<int> > results;

            partial_edge_set_type neighbors =
                lcos::eager_future<vertex::out_edges_action>(target).get();

            LSSCA_(info) << "Visiting " << neighbors.size()
                         << " neighbors of target " << target;

            partial_edge_set_type::iterator end = neighbors.end();
            for (partial_edge_set_type::iterator it = neighbors.begin();
                 it != end; ++it)
            {
                // Spawn subsequent search
                results.push_back(lcos::eager_future<
                    ssca2::extract_subgraph_action
                >(locale, H, pmap, target, (*it).target_, d-1));
            }

            // Collect notifications of when subsequent searches are finished
            std::vector<lcos::future_value<int> >::iterator rend = results.end();
            for (std::vector<lcos::future_value<int> >::iterator rit = results.begin();
                 rit != rend; ++rit)
            {
                (*rit).get();
            }
        }

        return 0;
    }

    int
    ssca2::init_props_map(naming::id_type P, naming::id_type G)
    {
        LSSCA_(info) << "Initializing property map";

        // Get vertex set of G
        typedef components::distributing_factory::result_type result_type;
        result_type vertices =
            lcos::eager_future<graph::vertices_action>(G).get();

        // Spawn inits on local sub lists
        result_type::const_iterator end = vertices.end();
        for (result_type::const_iterator it = vertices.begin();
             it != end; ++it)
        {
            naming::id_type local_props =
                lcos::eager_future<
                    dist_gids_map_type::get_local_action
                >(P, (*it).prefix_).get();
            lcos::eager_future<
                init_props_map_local_action
            >((*it).prefix_, local_props, *it).get();
        }

        return 0;
    }

    int
    ssca2::init_props_map_local(naming::id_type local_props,
                                locality_result local_vertices)
    {
        LSSCA_(info) << "Initializing local property map";

        // Not implemented ... yet.

        return 0;
    }


}}}

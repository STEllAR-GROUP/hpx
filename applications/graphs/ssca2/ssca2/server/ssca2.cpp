//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <fstream>
#include <algorithm>

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

#include <boost/unordered_map.hpp>

///////////////////////////////////////////////////////////////////////////////

typedef hpx::components::server::ssca2 ssca2_type;

typedef hpx::components::server::vertex vertex_type;
typedef hpx::components::server::distributed_set<vertex_type> dist_vertex_set_type;
typedef hpx::components::server::local_set<vertex_type> local_vertex_set_type;

typedef hpx::components::server::edge edge_type;
typedef hpx::components::server::distributed_set<edge_type> dist_edge_set_type;
typedef hpx::components::server::local_set<edge_type> local_edge_set_type;

typedef hpx::components::server::graph graph_type;
typedef hpx::components::server::distributed_set<graph_type> dist_graph_set_type;
typedef hpx::components::server::local_set<graph_type> local_graph_set_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the ssca2 actions
HPX_REGISTER_ACTION_EX(ssca2_type::read_graph_action,
                       ssca2_large_set_action);
HPX_REGISTER_ACTION_EX(ssca2_type::large_set_action,
                       ssca2_large_set_action);
HPX_REGISTER_ACTION_EX(ssca2_type::large_set_local_action,
                       ssca2_large_set_local_action);
HPX_REGISTER_ACTION_EX(ssca2_type::init_props_map_action,
                       ssca2_init_props_map_action);
HPX_REGISTER_ACTION_EX(ssca2_type::init_props_map_local_action,
                       ssca2_init_props_map_local_action);
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<ssca2_type>, ssca2);
HPX_DEFINE_GET_COMPONENT_TYPE(ssca2_type);

#define LSSCA_(lvl) LAPP_(lvl) << " [SSCA] "

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ssca2::ssca2() {
        LSSCA_(info) << "event: action(ssca2::ssca2) status(begin)";
        LSSCA_(info) << "parent(" << threads::get_parent_id() << ")";
    }
    
    int
    ssca2::read_graph(naming::id_type G, std::string filename)
    {
        LSSCA_(info) << "event: action(ssca2::read_graph) status(begin)";
        LSSCA_(info) << "parent(" << threads::get_parent_id() << ")";

        int64_t x, y, w;
        boost::unordered_map<int64_t, naming::id_type> known_vertices;
        //boost::unordered_map<int64_t, std::vector<int64_t> > known_edges;
        std::vector<lcos::future_value<int> > results;

        LSSCA_(info) << "read_graph(" << G << ", " << filename << ")";

        std::ifstream fin(filename.c_str());
        if (!fin)
        {
            std::cerr << "Error: could not open " << filename << std::endl;
            exit(1);
        }

        fin >> x;
        fin >> y;
        fin >> w;

        int num_edges_added = 0;
        while (!fin.eof())
        {
            LSSCA_(info) << num_edges_added << ": " << "Adding edge ("
                         << x << ", " << y << ", " << w
                         << ")";

            // Use hash to catch duplicate vertices
            // Note: Update this to do the two actions in parallel
            if (known_vertices.find(x) == known_vertices.end())
            {
                known_vertices[x] = lcos::eager_future<
                    server::graph::add_vertex_action
                >(G, naming::invalid_id).get();

                LSSCA_(info) << "Adding vertex (" << x << ", " << known_vertices[x] << ")";
            }
            if (known_vertices.find(y) == known_vertices.end())
            {
                known_vertices[y] = lcos::eager_future<
                    server::graph::add_vertex_action
                >(G, naming::invalid_id).get();

                LSSCA_(info) << "Adding vertex (" << y << ", " << known_vertices[y] << ")";
            }

            results.push_back(
                lcos::eager_future<
                    server::graph::add_edge_action
                >(G, known_vertices[x], known_vertices[y], w));

            num_edges_added += 1;

            fin >> x;
            fin >> y;
            fin >> w;
        }

        // Check that all in flight actions have completed
        while (results.size() > 0)
        {
            results.back().get();
            results.pop_back();
        }
    }

    int
    ssca2::large_set(naming::id_type G, naming::id_type dist_edge_set)
    {
        LSSCA_(info) << "event: action(ssca2::large_set) status(begin)";
        LSSCA_(info) << "parent(" << threads::get_parent_id() << ")";

        LSSCA_(info) << "large_set(" << G << ", " << dist_edge_set << ")";

        typedef components::distributing_factory::result_type result_type;
        typedef std::vector<naming::id_type> gids_type;

        int total_added = 0;
        std::vector<lcos::future_value<int> > local_searches;

        // Get vertex set of G
        naming::id_type vertices =
            lcos::eager_future<graph::vertices_action>(G).get();

        typedef std::vector<naming::id_type> gids_type;
        gids_type vertex_sets =
            lcos::eager_future<dist_vertex_set_type::locals_action>(vertices).get();

        // Create lcos to reduce max value
        lcos::reduce_max local_max(vertex_sets.size(), 1);
        lcos::reduce_max global_max(1, vertex_sets.size());

        // Spawn searches on local sub lists
        gids_type::const_iterator vend = vertex_sets.end();
        for (gids_type::const_iterator vit = vertex_sets.begin();
             vit != vend; ++vit)
        {
            naming::id_type there(boost::uint64_t((*vit).get_msb()) << 32, 0);

            local_searches.push_back(
                lcos::eager_future<large_set_local_action>(
                    there,
                    (*vit),
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
    ssca2::large_set_local(naming::id_type local_vertex_set,
                             naming::id_type edge_set,
                             naming::id_type local_max_lco,
                             naming::id_type global_max_lco)
    {
        LSSCA_(info) << "event: action(ssca2::large_set_local) status(begin)";
        LSSCA_(info) << "parent(" << threads::get_parent_id() << ")";

        LSSCA_(info) << "large_set_local(" << local_vertex_set << ", " << edge_set
                     << ", " << local_max_lco << ", " << global_max_lco << ")";

        typedef std::vector<naming::id_type> gids_type;

        int max = -1;
        int num_added = 0;

        naming::id_type here = applier::get_applier().get_runtime_support_gid();

        std::vector<edge_type::edge_snapshot_type> edge_set_local;

        // Iterate over local vertices
        gids_type vertices =
            lcos::eager_future<local_vertex_set_type::get_action>(local_vertex_set).get();

        gids_type::const_iterator vend = vertices.end();
        for (gids_type::const_iterator vit = vertices.begin();
             vit != vend; ++vit)
        {
            // Get incident edges from this vertex
            // Need to update this to be a guaranteed local action
            typedef vertex::partial_edge_set_type partial_type;
            partial_type partials =
                lcos::eager_future<vertex::out_edges_action>(*vit).get();

            // Iterate over incident edges
            // Could break this out into a separate thread to run concurrently
            partial_type::iterator pend = partials.end();
            for (partial_type::iterator pit = partials.begin(); pit != pend; ++pit)
            {
                if ((*pit).label_ > max)
                {
                    edge_set_local.clear();

                    edge_type::edge_snapshot_type e(*vit, (*pit).target_, (*pit).label_);

                    edge_set_local.push_back(e);

                    max = (*pit).label_;
                }
                else if ((*pit).label_ == max)
                {
                    edge_set_local.push_back(
                        edge::edge_snapshot_type(*vit, (*pit).target_, (*pit).label_));
                }
            }
        }

        LSSCA_(info) << "Max on locality " << here << " "
                    << "is " << max << ", total of " << edge_set_local.size();

        // Signal local maximum
        applier::apply<
            lcos::detail::reduce_max::signal_action
        >(local_max_lco, max);

        LSSCA_(info) << "Waiting to see max is on locality "
                    << here;

        // Wait for global maximum
        int global_max =
            lcos::eager_future<
                hpx::lcos::detail::reduce_max::wait_action
            >(global_max_lco).get();

        // Add local max edge set if it is globally maximal
        if (max == global_max)
        {
            LSSCA_(info) << "Adding local edge set at "
                        << here;

            naming::id_type local_set =
                lcos::eager_future<
                    dist_edge_set_type::get_local_action
                >(edge_set, here).get();

            naming::id_type edge_base(stubs::edge::create(here, edge_set_local.size()));
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
                        << here;
        }

        return num_added;
    }

    int
    ssca2::init_props_map(naming::id_type P, naming::id_type G)
    {
        LSSCA_(info) << "Initializing property map";

        // Action intentionally left blank

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

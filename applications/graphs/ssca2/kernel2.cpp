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

#include <hpx/components/distributing_factory/distributing_factory.hpp>

#include <hpx/components/graph/graph.hpp>
#include <hpx/components/graph/edge.hpp>
#include <hpx/components/graph/vertex.hpp>

#include <hpx/components/distributed_set/distributed_set.hpp>
#include <hpx/components/distributed_set/local_set.hpp>
#include <hpx/components/distributed_set/server/distributed_set.hpp>
#include <hpx/components/distributed_set/server/local_set.hpp>

#include <hpx/components/distributed_map/distributed_map.hpp>
#include <hpx/components/distributed_map/local_map.hpp>

#include "ssca2_benchmark.hpp"

#include "props/props.hpp"
#include "reduce_max/reduce_max.hpp"

#include <boost/unordered_map.hpp>

using namespace hpx;

#define LSSCA_(lvl) LAPP_(lvl) << " [SSCA] "

///////////////////////////////////////////////////////////////////////////////

HPX_REGISTER_ACTION(kernel2_action);

int kernel2(naming::id_type G, naming::id_type dist_edge_set)
{
    LSSCA_(info) << "large_set(" << G << ", " << dist_edge_set << ")";

    typedef distributing_factory_type::result_type result_type;

    int total_added = 0;
    future_ints_type local_searches;

    // Get vertex set of G
    gid_type vertices =
        lcos::eager_future<graph_type::vertices_action>(G).get();

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
        gid_type there =
            lcos::eager_future<local_vertex_set_type::get_locale_action>(*vit).get();

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

HPX_REGISTER_ACTION(large_set_local_action);

int large_set_local(naming::id_type local_vertex_set,
                         naming::id_type edge_set,
                         naming::id_type local_max_lco,
                         naming::id_type global_max_lco)
{
    LSSCA_(info) << "large_set_local(" << local_vertex_set << ", " << edge_set
                 << ", " << local_max_lco << ", " << global_max_lco << ")";

    int max = -1;
    int num_added = 0;

    naming::id_type here = find_here();

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
        typedef vertex_type::partial_edge_set_type partial_type;
        partial_type partials =
            lcos::eager_future<vertex_type::out_edges_action>(*vit).get();

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
                    edge_type::edge_snapshot_type(*vit, (*pit).target_, (*pit).label_));
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

        gid_type local_set =
            lcos::eager_future<
                dist_edge_set_type::get_local_action
            >(edge_set, here).get();

        gid_type edge_base(stub_edge_type::create(here, edge_set_local.size()));
        gids_type edges(edge_set_local.size());

        int i=0;
        std::vector<edge_type::edge_snapshot_type>::const_iterator eend = edge_set_local.end();
        for (std::vector<edge_type::edge_snapshot_type>::const_iterator eit = edge_set_local.begin();
             eit != eend; ++eit, ++i)
        {
            lcos::eager_future<
                           edge_type::init_action
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

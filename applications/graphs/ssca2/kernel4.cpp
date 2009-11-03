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

HPX_REGISTER_ACTION(kernel4_action);

int kernel4(naming::id_type V, naming::id_type VS, int k4_approx, naming::id_type bc_scores)
{
    future_ints_type results;

    // Build out BC pmap
    gids_type locals =
        lcos::eager_future<dist_vertex_set_type::locals_action>(V).get();

    // This uses hack to get prefix
    naming::id_type there(boost::uint64_t((locals[0]).get_msb()) << 32, 0);

    gids_type::const_iterator lit = locals.begin();
    for (gids_type::const_iterator lend = locals.end();
         lit != lend; ++lit)
    {
        // This uses hack to get prefix
        naming::id_type there(boost::uint64_t((*lit).get_msb()) << 32, 0);

        gid_type bc_local =
            lcos::eager_future<dist_gids_map_type::get_local_action>(bc_scores, there).get();

        results.push_back(lcos::eager_future<init_local_bc_action>(there, bc_local, *lit));
    }
    while (results.size() > 0)
    {
        results.back().get();
        results.pop_back();
    }

    // Do parallel BFS/SSSP's
    gids_type vs_locals = lcos::eager_future<dist_vertex_set_type::locals_action>(VS).get();
    gids_type::const_iterator vsit = vs_locals.begin();
    for (gids_type::const_iterator vsend = vs_locals.end();
         vsit != vsend; ++vsit)
    {
        // This uses hack to get prefix
        naming::id_type there(boost::uint64_t((*vsit).get_msb()) << 32, 0);

        results.push_back(lcos::eager_future<bfs_sssp_local_action>(there, V, *vsit, bc_scores));
    }
    while(results.size() > 0)
    {
        results.back().get();
        results.pop_back();
    }

    return 0;
}

void select_random_vertices(gids_type v_locals, int k4_approx, naming::id_type VS)
{
    LSSCA_(info) << "Selecting random vertices.";

    boost::unordered_map<int, int> seen_vertices;

    std::vector<int> sizes(v_locals.size());
    int total = 0;

    gids_type vs_locals(v_locals.size());

    for (int i = 0; i < sizes.size(); ++i)
    {
        sizes[i] = lcos::eager_future<local_vertex_set_type::size_action>(v_locals[i]).get();
        total += sizes[i];

        // This uses hack to get prefix
        naming::id_type there(boost::uint64_t((v_locals[i]).get_msb()) << 32, 0);
        vs_locals[i] = lcos::eager_future<dist_vertex_set_type::get_local_action>(VS, there).get();

        LSSCA_(info) << "size[" << i << "] = " << sizes[i];
    }

    // Randomly select 2^k4_approx vertices
    // Note: replace any zero-degree vertices
    srand(time(0));
    int num_to_add = 1 << k4_approx;
    while (num_to_add > 0)
    {
        LSSCA_(info) << "Attempting to add " << num_to_add;

        //for (int i=0; i<num_to_add; ++i)
        while (num_to_add > 0)
        {
            int index = rand()*(1.0*total)/(RAND_MAX);

            if (seen_vertices.find(index) == seen_vertices.end())
            {
                int i = 0;
                int sum = 0;
                while (index < sum + sizes[i])
                {
                    sum += sizes[i];
                    ++i;
                }
                --i;

                int local_index = index - (sum - sizes[i]);

               // This uses hack to get prefix
               naming::id_type there(boost::uint64_t((v_locals[i]).get_msb()) << 32, 0);

               // Need to parallelize this part ...
               int added = lcos::eager_future<add_local_item_action>(there, local_index, v_locals[i], vs_locals[i]).get();

               if (added == 1)
               {
                   seen_vertices[index] = 1;
                   num_to_add -= 1;

               }
            }
        }
    }

    return;
}

double calculate_teps(gid_type V, int order, double total_time, bool exact)
{
    // Count number of zero-degree vertices
    int n_0 = 0;

    gids_type v_locals = components::stubs::distributed_set<vertex_type>::locals(V);
    gids_type::const_iterator vit = v_locals.begin();
    for (gids_type::const_iterator vend = v_locals.end();
         vit != vend; ++vit)
    {
        gids_type vertices = components::stubs::local_set<vertex_type>::get(*vit);
        gids_type::const_iterator it = vertices.begin();
        for (gids_type::const_iterator end = vertices.end();
             it != end; ++it)
        {
            gid_type v = *it;
            int deg_v = components::stubs::vertex::out_degree(v);

            if (deg_v == 0)
            {
                ++n_0;
            }
            LSSCA_(info) << "Deg, " << v << ", " << deg_v;
        }
    }

    // Calcuate TEPS
    double teps;

    if (exact)
    {
        teps = (7 * order * (order-n_0)) / total_time;
    }
    else
    {
        teps = (7 * order * (order)) / total_time;
    }

    return teps;
}

///////////////////////////////////////////////////////////////////////////////
// def init_local_bc(local_map bc_local, local_set v_local):
//   For each v in v_local:
//     bc_local[v] = 0

HPX_REGISTER_ACTION(init_local_bc_action);

int init_local_bc(naming::id_type bc_local, naming::id_type v_local)
{
    LSSCA_(info) << "Initializing local_bc (" << bc_local << ") with v_local (" << v_local << ")";

    gids_type vertices = lcos::eager_future<local_vertex_set_type::get_action>(v_local).get();

    gids_type::const_iterator vit = vertices.begin();
    for (gids_type::const_iterator vend = vertices.end();
         vit != vend; ++vit)
    {
        components::component_type props_comp_type = components::get_component_type<props_type>();
        lcos::eager_future<local_gids_map_type::value_action>(bc_local, *vit, props_comp_type).get();
    }

    return 0;
}

///////////////////////////////////////////////////////////////////////////////

HPX_REGISTER_ACTION(add_local_item_action);

int add_local_item(int index, naming::id_type v_locals, naming::id_type vs_locals)
{
    gids_type vertices = lcos::eager_future<local_vertex_set_type::get_action>(v_locals).get();

    int retval = 0;

    int degree = lcos::eager_future<vertex_type::out_degree_action>(vertices[index]).get();
    if (degree > 0)
    {
        lcos::eager_future<local_vertex_set_type::add_item_action>(vs_locals, vertices[index]).get();
        retval = 1;
    }

    return retval;
}

///////////////////////////////////////////////////////////////////////////////

HPX_REGISTER_ACTION(bfs_sssp_local_action);

int bfs_sssp_local(naming::id_type V, naming::id_type vs_locals, naming::id_type bc_scores)
{
    LSSCA_(info) << "bfs_sssp_local(" << vs_locals << ", " << bc_scores << ")";

    future_ints_type results;

    gid_type here = applier::get_applier().get_prefix();

    gids_type vertices = lcos::eager_future<local_vertex_set_type::get_action>(vs_locals).get();
    gids_type::const_iterator vit = vertices.begin();
    for (gids_type::const_iterator vend = vertices.end();
         vit != vend; ++vit)
    {
        results.push_back(lcos::eager_future<bfs_sssp_action>(here, *vit, V, bc_scores));
    }
    while (results.size() > 0)
    {
        results.back().get();
        results.pop_back();
    }

    return 0;

}

///////////////////////////////////////////////////////////////////////////////
// This is not very parallel

HPX_REGISTER_ACTION(bfs_sssp_action);

int bfs_sssp(naming::id_type start, naming::id_type V, naming::id_type bc_scores)
{
    LSSCA_(info) << "bfs_sssp(" << start << ", " << V << ", " << bc_scores << ")";

    gids_type S;
    boost::unordered_map<gid_type, gids_type> P;
    boost::unordered_map<gid_type, int> sigma;
    boost::unordered_map<gid_type, int> d;
    boost::unordered_map<gid_type, double> delta;
    std::queue<gid_type> Q;

    // Build out P, sigma, d, and delta over V
    gids_type v_locals = lcos::eager_future<dist_vertex_set_type::locals_action>(V).get();
    gids_type::const_iterator vlit = v_locals.begin();
    for (gids_type::const_iterator vlend = v_locals.end();
         vlit != vlend; ++vlit)
    {
        gids_type vertices = lcos::eager_future<local_vertex_set_type::get_action>(*vlit).get();
        gids_type::const_iterator vit = vertices.begin();
        for (gids_type::const_iterator vend = vertices.end();
             vit != vend; ++vit)
        {
            // Each v in V
            gid_type v = *vit;
            P[v] = gids_type();
            sigma[v] = 0;
            d[v] = -1;
            delta[v] = 0.0;
        }
    }

    // Initialze start values
    sigma[start] = 1;
    d[start] = 0;
    Q.push(start);

    while (!Q.empty())
    {
        gid_type v = Q.front();
        Q.pop();

        S.push_back(v);

        // Non-parallel for-loop
        vertex_type::partial_edge_set_type out_edges = lcos::eager_future<vertex_type::out_edges_action>(v).get();
        vertex_type::partial_edge_set_type::const_iterator oit = out_edges.begin();
        for (vertex_type::partial_edge_set_type::const_iterator oend = out_edges.end();
             oit != oend; ++oit)
        {
            // Filter edges
            if (oit->label_ & 7 != 0)
            {
                gid_type w = oit->target_;

                if (d[w] < 0)
                {
                    Q.push(w);
                    d[w] = d[v] + 1;
                }

                if (d[w] = d[v] + 1)
                {
                    sigma[w] = sigma[w] + sigma[v];
                    P[w].push_back(v);
                }
            }
        }

        while (!S.empty())
        {
            gid_type w = S.back();
            S.pop_back();

            LSSCA_(info) << "Looking at w = " << w;

            gids_type::const_iterator pit = P[w].begin();
            for (gids_type::const_iterator pend = P[w].end();
                 pit != pend; ++pit)
            {
                gid_type v = *pit;

                LSSCA_(info) << "\tLooking at v = " << v;

                LSSCA_(info) << "\tdelta[v] = " << delta[v]
                          << " sigma[v] = " << sigma[v]
                          << " sigma[w] = " << sigma[w]
                          << " delta[w] = " << delta[w];

                delta[v] = delta[v] + (1.0*sigma[v]/sigma[w])*(1 + delta[w]);

                LSSCA_(info) << "\tnew delta[v] = " << delta[v];
            }
        }
    }

    future_ints_type results;
    boost::unordered_map<gid_type, double>::const_iterator ait = delta.begin();
    for (boost::unordered_map<gid_type, double>::const_iterator aend = delta.end();
         ait != aend; ++ait)
    {
        gid_type w = ait->first;

        // This uses hack to get prefix
        naming::id_type there(boost::uint64_t((w).get_msb()) << 32, 0);

        // Should this be divided by 2?
        results.push_back(lcos::eager_future<incr_bc_action>(there, bc_scores, w, delta[w]/2.0));

    }
    while (results.size() > 0)
    {
        results.back().get();
        results.pop_back();
    }

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
// def init_local_bc(local_map bc_local, local_set v_local):
//   For each v in v_local:
//     bc_local[v] = 0

HPX_REGISTER_ACTION(incr_bc_action);

int incr_bc(naming::id_type bc_scores, naming::id_type w, double delta_w)
{
    gid_type here = applier::get_applier().get_prefix();

    gid_type bc_local = lcos::eager_future<dist_gids_map_type::get_local_action>(bc_scores, here).get();

    components::component_type props_comp_type = components::get_component_type<props_type>();
    gid_type bc_w_prop = lcos::eager_future<local_gids_map_type::value_action>(bc_local, w, props_comp_type).get();
    int bc_w = lcos::eager_future<props_type::incr_action>(bc_w_prop, delta_w).get();

    LSSCA_(info) <<  "BC, " << w <<", " << delta_w;

    return 0;
}

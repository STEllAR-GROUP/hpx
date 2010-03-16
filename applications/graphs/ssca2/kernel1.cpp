//  Copyright (c) 2009-2010 Dylan Stark
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <fstream>
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
#include <boost/tuple/tuple.hpp>

using namespace hpx;

#define LSSCA_(lvl) LAPP_(lvl) << " [SSCA] "

///////////////////////////////////////////////////////////////////////////////
// Kernel 1 v 1
// - Synchronous vertex addition
// - Asynchronous edge insertion

HPX_REGISTER_ACTION(kernel1_v1_action);

int kernel1_v1(naming::id_type G, std::string filename)
{
    hpx::util::high_resolution_timer k1_t;

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
                graph_type::add_vertex_action
            >(G, naming::invalid_id).get();

            LSSCA_(info) << "Adding vertex (" << x << ", " << known_vertices[x] << ")";
        }
        if (known_vertices.find(y) == known_vertices.end())
        {
            known_vertices[y] = lcos::eager_future<
                graph_type::add_vertex_action
            >(G, naming::invalid_id).get();

            LSSCA_(info) << "Adding vertex (" << y << ", " << known_vertices[y] << ")";
        }

        results.push_back(
            lcos::eager_future<
                graph_type::add_edge_action
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

    std::cout << "Internal Kernel 1 v1 in " << k1_t.elapsed() << " sec" << std::endl;

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
// Kernel 1 v 2
// - Asynchronous vertex addition
// - Asynchronous edge insertion
// - Two Phase

HPX_REGISTER_ACTION(kernel1_v2_action);

int kernel1_v2(naming::id_type G, std::string filename)
{
    hpx::util::high_resolution_timer k1_t;

    int64_t x, y, w;
    boost::unordered_map<int64_t, int> known_vertices;
    future_gids_type vertex_results;
    future_ints_type edge_results;

    typedef boost::tuple<int64_t, int64_t, int64_t> edge_tuple_type;
    std::vector<edge_tuple_type> edges;

    gids_type vertices;

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

    // Phase 1: asynchronous vertex addition
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
            vertex_results.push_back(lcos::eager_future<
                graph_type::add_vertex_action
            >(G, naming::invalid_id));
            known_vertices[x] = vertex_results.size()-1;

            LSSCA_(info) << "Adding vertex (" << x << ", " << x << ")";
        }
        if (known_vertices.find(y) == known_vertices.end())
        {
            vertex_results.push_back(lcos::eager_future<
                graph_type::add_vertex_action
            >(G, naming::invalid_id));
            known_vertices[y] = vertex_results.size()-1;

            LSSCA_(info) << "Adding vertex (" << y << ", " << y << ")";
        }

        edges.push_back(edge_tuple_type(x,y,w));

        fin >> x;
        fin >> y;
        fin >> w;
    }
    while (vertex_results.size() > 0)
    {
        vertices.push_back(vertex_results.back().get());
        vertex_results.pop_back();
    }

    // Phase 2: asynchronous edge insertion
    while (edges.size() > 0)
    {
        edge_tuple_type e = edges.back();
        edges.pop_back();

        edge_results.push_back(
            lcos::eager_future<
                graph_type::add_edge_action
            >(G,
              vertices[known_vertices[e.get<0>()]],
              vertices[known_vertices[e.get<1>()]],
              e.get<2>()));

        num_edges_added += 1;
    }
    while (edge_results.size() > 0)
    {
        edge_results.back().get();
        edge_results.pop_back();
    }

    std::cout << "Internal Kernel 1 v2 in " << k1_t.elapsed() << " sec" << std::endl;

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
// Kernel 1 v 3
// - Asynchronous vertex addition
// - Asynchronous edge insertion
// - One Phase

HPX_REGISTER_ACTION(kernel1_v3_action);

int kernel1_v3(naming::id_type G, std::string filename)
{
    hpx::util::high_resolution_timer k1_t;

    int64_t x, y, w;
    boost::unordered_map<int64_t, int> known_vertices;
    future_gids_type vertex_results;
    future_ints_type edge_results;

    typedef boost::tuple<int64_t, int64_t, int64_t> edge_tuple_type;
    std::vector<edge_tuple_type> edges;

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
            vertex_results.push_back(lcos::eager_future<
                graph_type::add_vertex_action
            >(G, naming::invalid_id));
            known_vertices[x] = vertex_results.size()-1;

            LSSCA_(info) << "Adding vertex (" << x << ", " << x << ")";
        }
        if (known_vertices.find(y) == known_vertices.end())
        {
            vertex_results.push_back(lcos::eager_future<
                graph_type::add_vertex_action
            >(G, naming::invalid_id));
            known_vertices[y] = vertex_results.size()-1;

            LSSCA_(info) << "Adding vertex (" << y << ", " << y << ")";
        }

        edges.push_back(edge_tuple_type(x,y,w));

        fin >> x;
        fin >> y;
        fin >> w;
    }

    std::vector<edge_tuple_type>::const_iterator eend = edges.end();
    for (std::vector<edge_tuple_type>::const_iterator eit = edges.begin();
         eit != eend; ++eit)
    {
        edge_results.push_back(
            lcos::eager_future<
                graph_type::add_edge_action
            >(G,
              vertex_results[known_vertices[eit->get<0>()]].get(),
              vertex_results[known_vertices[eit->get<1>()]].get(),
              eit->get<2>()));

        num_edges_added += 1;
    }
    while (edge_results.size() > 0)
    {
        edge_results.back().get();
        edge_results.pop_back();
    }

    std::cout << "Internal Kernel 1 v3 in " << k1_t.elapsed() << " sec" << std::endl;

    return 0;
}

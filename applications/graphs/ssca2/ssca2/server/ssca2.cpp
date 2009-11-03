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
                       ssca2_read_graph_action);
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

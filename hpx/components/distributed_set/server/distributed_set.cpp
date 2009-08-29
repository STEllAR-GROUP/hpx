//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include <hpx/components/vertex/vertex.hpp>
#include <hpx/components/graph/edge.hpp>
#include <hpx/components/graph/graph.hpp>

#include "distributed_set.hpp"
#include "../stubs/distributed_set.hpp"

#include "../local_set.hpp"
#include "../server/local_set.hpp"
#include "../stubs/local_set.hpp"

// Needs this to define edge_set_type

///////////////////////////////////////////////////////////////////////////////

typedef hpx::components::server::distributed_set<
    hpx::components::vertex
> distributed_vertex_set_type;

typedef hpx::components::server::distributed_set<
    hpx::components::edge
> distributed_edge_set_type;

typedef hpx::components::server::distributed_set<
    hpx::components::graph
> distributed_graph_set_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the distributed_set actions

HPX_REGISTER_ACTION_EX(
    distributed_vertex_set_type::get_local_action,
    distributed_vertex_set_get_local_action);
HPX_REGISTER_ACTION_EX(
    distributed_vertex_set_type::locals_action,
    distributed_vertex_set_locals_action);
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<distributed_vertex_set_type>,
    distributed_vertex_set);
HPX_DEFINE_GET_COMPONENT_TYPE(distributed_vertex_set_type);

HPX_REGISTER_ACTION_EX(
    distributed_edge_set_type::get_local_action,
    distributed_edge_set_get_local_action);
HPX_REGISTER_ACTION_EX(
    distributed_edge_set_type::locals_action,
    distributed_edge_set_locals_action);
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<distributed_edge_set_type>,
    distributed_edge_set);
HPX_DEFINE_GET_COMPONENT_TYPE(distributed_edge_set_type);

HPX_REGISTER_ACTION_EX(
    distributed_graph_set_type::get_local_action,
    distributed_graph_set_get_local_action);
HPX_REGISTER_ACTION_EX(
    distributed_graph_set_type::locals_action,
    distributed_graph_set_locals_action);
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<distributed_graph_set_type>,
    distributed_graph_set);
HPX_DEFINE_GET_COMPONENT_TYPE(distributed_graph_set_type);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    template <typename Item>
    distributed_set<Item>::distributed_set()
      : next_locale_(0)
    {}
    
    template <typename Item>
    naming::id_type distributed_set<Item>::get_local(naming::id_type locale)
    {
        std::cout << "Getting local sublist at " << locale <<  std::endl;

        // If a list already exists on this locality, return a reference to it
        // Otherwise, create a new one and return a reference to it
        if (map_.count(locale) == 0)
        {
            std::cout << "Need to make new sublist" << std::endl;

            // Create a new sub list there
            typedef hpx::components::local_set<Item> local_set_type;
            local_set_type edge_set(local_set_type::create(locale));

            std::cout << "Created local sublist" << std::endl;

            map_[locale] = edge_set.get_gid();
            locals_.push_back(map_[locale]);

            std::cout << "Sending local sublist gid" << std::endl;

            return map_[locale];
        }
        else
        {
            return map_[locale];
        }
    }

    template <typename Item>
    std::vector<naming::id_type> distributed_set<Item>::locals(void)
    {
        std::cout << "Getting coverage of distributed set" << std::endl;

        return locals_;
    }

}}}

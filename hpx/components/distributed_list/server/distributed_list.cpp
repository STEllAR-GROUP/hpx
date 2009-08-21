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

#include "distributed_list.hpp"
#include "../stubs/distributed_list.hpp"

#include "../local_list.hpp"
#include "../server/local_list.hpp"
#include "../stubs/local_list.hpp"

// Needs this to define edge_list_type
#include "../../../../applications/graphs/ssca2/ssca2/ssca2.hpp"

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::server::distributed_list<
    hpx::components::server::ssca2::edge_list_type
> distributed_edge_list_type;

typedef hpx::components::server::distributed_list<
    hpx::components::server::ssca2::graph_list_type
> distributed_graph_list_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the distributed_list actions

HPX_REGISTER_ACTION_EX(
    distributed_edge_list_type::get_local_action,
    distributed_edge_list_get_local_action);
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<distributed_edge_list_type>,
    distributed_edge_list);
HPX_DEFINE_GET_COMPONENT_TYPE(distributed_edge_list_type);

HPX_REGISTER_ACTION_EX(
    distributed_graph_list_type::get_local_action,
    distributed_graph_list_get_local_action);
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<distributed_graph_list_type>,
    distributed_graph_list);
HPX_DEFINE_GET_COMPONENT_TYPE(distributed_graph_list_type);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    template <typename List>
    distributed_list<List>::distributed_list() {}
    
    template <typename List>
    naming::id_type distributed_list<List>::get_local(naming::id_type locale)
    {
        std::cout << "Getting local sublist at " << locale <<  std::endl;

        // If a list already exists on this locality, return a reference to it
        // Otherwise, create a new one and return a reference to it
        if (map_.count(locale) == 0)
        {
            std::cout << "Need to make new sublist" << std::endl;

            // Create a new sub list there
            typedef ssca2::edge_list_type edge_list_type;
            typedef hpx::components::local_list<edge_list_type> local_list_type;

            local_list_type edge_list(local_list_type::create(locale));

            std::cout << "Created local sublist" << std::endl;

            map_[locale] = edge_list.get_gid();

            std::cout << "Sending local sublist gid" << std::endl;

            return map_[locale];
        }
        else
        {
            return map_[locale];
        }
    }

}}}

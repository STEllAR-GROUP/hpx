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

#include "local_set.hpp"
#include "../stubs/local_set.hpp"

// Needs this to define edge_set_type
#include "../../../../applications/graphs/ssca2/ssca2/ssca2.hpp"

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::server::local_set<
    hpx::components::server::ssca2::edge_set_type
> local_edge_set_type;

typedef hpx::components::server::local_set<
    hpx::components::server::ssca2::graph_set_type
> local_graph_set_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the local_set actions

HPX_REGISTER_ACTION_EX(
    local_edge_set_type::append_action,
    local_set_append_action);
HPX_REGISTER_ACTION_EX(
    local_edge_set_type::get_action,
    local_set_get_action);
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<local_edge_set_type>, local_set);
HPX_DEFINE_GET_COMPONENT_TYPE(local_edge_set_type);

HPX_REGISTER_ACTION_EX(
    local_graph_set_type::append_action,
    local_set_append_action);
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<local_graph_set_type>, local_graph_set);
HPX_DEFINE_GET_COMPONENT_TYPE(local_graph_set_type);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    template <typename List>
    local_set<List>::local_set()
      : local_set_(0)
    {}

    template <typename List>
    int local_set<List>::append(List list)
    {
        std::cout << "Appending to local list at locale " << std::endl;

        // Probably should do some locking ... somewhere ... maybe here

        typedef typename List::iterator list_iter;
        list_iter end = list.end();
        for (list_iter it = list.begin(); it != end; ++it)
        {
            local_set_.push_back(*it);
        }

        return local_set_.size();
    }

    template <typename List>
    List local_set<List>::get(void)
    {
        std::cout << "Getting local set" << std::endl;

        return local_set_;
    }
}}}

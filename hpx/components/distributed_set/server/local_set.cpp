//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include <hpx/components/vertex/vertex.hpp>
#include <hpx/components/graph/graph.hpp>
#include <hpx/components/graph/edge.hpp>

#include "local_set.hpp"
#include "../stubs/local_set.hpp"

#define LLSET_(lvl) LAPP_(lvl) << " [LOCAL_SET] " << gid_ << " "

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::server::local_set<
    hpx::components::server::vertex
> local_vertex_set_type;

typedef hpx::components::server::local_set<
    hpx::components::server::edge
> local_edge_set_type;

typedef hpx::components::server::local_set<
    hpx::components::server::graph
> local_graph_set_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the local_set actions

HPX_REGISTER_ACTION_EX(
    local_vertex_set_type::add_item_action,
    local_vertex_set_add_item_action);
HPX_REGISTER_ACTION_EX(
    local_vertex_set_type::append_action,
    local_vertex_set_append_action);
HPX_REGISTER_ACTION_EX(
    local_vertex_set_type::get_action,
    local_vertex_set_get_action);
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<local_vertex_set_type>, local_vertex_set);
HPX_DEFINE_GET_COMPONENT_TYPE(local_vertex_set_type);

HPX_REGISTER_ACTION_EX(
    local_edge_set_type::add_item_action,
    local_edge_set_add_item_action);
HPX_REGISTER_ACTION_EX(
    local_edge_set_type::append_action,
    local_edge_set_append_action);
HPX_REGISTER_ACTION_EX(
    local_edge_set_type::get_action,
    local_edge_set_get_action);
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<local_edge_set_type>, local_edge_set);
HPX_DEFINE_GET_COMPONENT_TYPE(local_edge_set_type);

HPX_REGISTER_ACTION_EX(
    local_graph_set_type::add_item_action,
    local_edge_set_add_item_action);
HPX_REGISTER_ACTION_EX(
    local_graph_set_type::append_action,
    local_edge_set_append_action);
HPX_REGISTER_ACTION_EX(
    local_graph_set_type::get_action,
    local_edge_set_get_action);
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<local_graph_set_type>, local_graph_set);
HPX_DEFINE_GET_COMPONENT_TYPE(local_graph_set_type);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    template <typename Item>
    local_set<Item>::local_set()
      : gid_(this->base_type::get_gid()),
        local_set_(0)
    {}

    template <typename Item>
    naming::id_type local_set<Item>::add_item(naming::id_type item=naming::invalid_id)
    {
        if (item == naming::invalid_id)
        {
            LLSET_(info) << "Adding new item";

            naming::id_type here = applier::get_applier().get_runtime_support_gid();

            components::component_type type = Item::get_component_type();

            item = components::stubs::runtime_support::create_component(here, type, 1);
        }
        else
        {
            LLSET_(info) << "Adding existing item";
        }

        if (item != naming::invalid_id)
        {
            // Just guard against concurrent updates
            lcos::mutex::scoped_lock l(local_set_mtx_);

            local_set_.push_back(item);
        }

        return item;
    }

    template <typename Item>
    int local_set<Item>::append(set_type list)
    {
        LLSET_(info) << "THIS IS DEPRICATED => DON'T USE";
        LLSET_(info) << "Appending to local list at locale ";

        // Probably should do some locking ... somewhere ... maybe here

        typedef typename set_type::iterator list_iter;
        list_iter end = list.end();
        for (list_iter it = list.begin(); it != end; ++it)
        {
            local_set_.push_back(*it);
        }

        return local_set_.size();
    }

    template <typename Item>
    std::vector<naming::id_type> local_set<Item>::get(void)
    {
        LLSET_(info)  << "Getting local set";

        return local_set_;
    }
}}}

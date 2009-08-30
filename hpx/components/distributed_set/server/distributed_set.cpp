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
#include <hpx/components/graph/edge.hpp>
#include <hpx/components/graph/graph.hpp>

#include "distributed_set.hpp"
#include "../stubs/distributed_set.hpp"

#include "../local_set.hpp"
#include "../server/local_set.hpp"
#include "../stubs/local_set.hpp"

#define LDSET_(lvl) LAPP_(lvl) << " [DIST_SET] " << gid_ << " "

///////////////////////////////////////////////////////////////////////////////

typedef hpx::components::server::distributed_set<
    hpx::components::server::vertex
> distributed_vertex_set_type;

typedef hpx::components::server::distributed_set<
    hpx::components::server::edge
> distributed_edge_set_type;

typedef hpx::components::server::distributed_set<
    hpx::components::server::graph
> distributed_graph_set_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the distributed_set actions

HPX_REGISTER_ACTION_EX(
    distributed_vertex_set_type::init_action,
    distributed_vertex_set_init_action);
HPX_REGISTER_ACTION_EX(
    distributed_vertex_set_type::add_item_action,
    distributed_vertex_set_add_item_action);
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
    distributed_edge_set_type::init_action,
    distributed_edge_set_init_action);
HPX_REGISTER_ACTION_EX(
    distributed_edge_set_type::add_item_action,
    distributed_edge_set_add_item_action);
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
    distributed_graph_set_type::init_action,
    distributed_graph_set_init_action);
HPX_REGISTER_ACTION_EX(
    distributed_graph_set_type::add_item_action,
    distributed_graph_set_add_item_action);
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
      : gid_(this->base_type::get_gid()),
        num_items_(0),
        next_locale_(0)
    {}
    
    template <typename Item>
    int distributed_set<Item>::init(int num_items)
    {
        LDSET_(info) << "DEPRICATED => DON'T YOU USE IT";
        LDSET_(info) << "distributed_set<>::init(" << num_items << ")";

        num_items_ = num_items;

        // Use a distributing factory for initial data distribution
        using components::distributing_factory;

        naming::id_type here = applier::get_applier().get_runtime_support_gid();
        distributing_factory factory(distributing_factory::create(here, true));

        typedef distributing_factory::result_type result_type;
        result_type list = factory.create_components(Item::get_component_type(), num_items);

        LDSET_(info) << "Number of sub lists: " << list.size();

        // Populate local sets
        std::vector<lcos::future_value<int> > results;
        result_type::const_iterator rend = list.end();
        for (result_type::const_iterator rit = list.begin();
             rit != rend; ++rit)
        {
            // Create a vector of GIDs from locality result
            std::vector<naming::id_type> gids;
            for (int i=0; i < (*rit).count_; ++i)
            {
                gids.push_back((*rit).first_gid_ + i);
            }

            // Append to appropriate local set
            naming::id_type there = get_local((*rit).prefix_); // Can I do this?

            results.push_back(
                lcos::eager_future<
                    typename local_set<Item>::append_action
                >(there, gids));

            LDSET_(info) << "Adding local set to locals_";
        }
        // Collect references to local sets
        while (results.size() > 0)
        {
            results.back().get();
            results.pop_back();
        }

        LDSET_(info) << "Finished initializing distributed set";

        return num_items;
    }

    template <typename Item>
    naming::id_type distributed_set<Item>::add_item(void)
    {
        LDSET_(info) << "add_item()";

        // Decide which locality to add it to
        // This should really use some DistributingPolicy interface

        // Part of this brought over from distributing_factor
        // make sure we get prefixes for derived component type, if any
        components::component_type type = Item::get_component_type();
        components::component_type prefix_type = type;
        if (type != components::get_base_type(type))
            prefix_type = components::get_derived_type(type);

        LDSET_(info) << "Got component type";

        // get list of locality prefixes
        std::vector<naming::id_type> prefixes;
        hpx::applier::get_applier().get_agas_client().get_prefixes(prefixes, prefix_type);

        if (prefixes.empty())
        {
            HPX_THROW_EXCEPTION(bad_component_type,
                "distributing_factory::create_components",
                "attempt to create component instance of unknown type: " +
                components::get_component_type_name(type));
        }

        LDSET_(info) << "Looking for locals_[" << num_items_+1 << "%" << prefixes.size() << " ="
                    << (num_items_+1)%prefixes.size() << "]";

        naming::id_type locale = get_local(prefixes[(num_items_+1)%prefixes.size()]);
        LDSET_(info) << "Adding new item to " << locale;

        // Add it there
        naming::id_type new_item =  lcos::eager_future<
                   typename components::server::local_set<Item>::add_item_action
               >(locale).get();

        if (new_item != naming::invalid_id)
        {
            lcos::mutex::scoped_lock l(mtx_);

            num_items_++;
        }

        return new_item;
    }

    template <typename Item>
    naming::id_type distributed_set<Item>::get_local(naming::id_type locale)
    {
        LDSET_(info) << "Getting local sublist at " << locale;

        // If a list already exists on this locality, return a reference to it
        // Otherwise, create a new one and return a reference to it
        if (map_.count(locale) == 0)
        {
            LDSET_(info) << "Need to make new sublist";

            // Create a new sub list there
            typedef hpx::components::local_set<Item> local_set_type;
            local_set_type edge_set(local_set_type::create(locale));

            LDSET_(info) << "Created local sublist";

            map_[locale] = edge_set.get_gid();
            locals_.push_back(map_[locale]);

            LDSET_(info) << "Sending local sublist gid";

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
        LDSET_(info) << "Getting coverage of distributed set";

        return locals_;
    }

}}}

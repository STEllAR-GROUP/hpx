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

#include "local_map.hpp"
#include "../stubs/local_map.hpp"

// Needs this to define edge_map_type
#include "../../../../applications/graphs/ssca2/ssca2/ssca2.hpp"

///////////////////////////////////////////////////////////////////////////////

typedef hpx::components::server::local_map<
    std::map<hpx::naming::id_type,hpx::naming::id_type>
> local_gids_map_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the local_map actions

HPX_REGISTER_ACTION_EX(
    local_gids_map_type::append_action,
    local_gids_map_append_action);
HPX_REGISTER_ACTION_EX(
    local_gids_map_type::value_action,
    local_gids_map_value_action);
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<local_gids_map_type>, local_gids_map);
HPX_DEFINE_GET_COMPONENT_TYPE(local_gids_map_type);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    template <typename List>
    local_map<List>::local_map()
      : local_map_()
    {}

    template <typename List>
    int local_map<List>::append(List list)
    {
        std::cout << "Appending to local list at locale " << std::endl;

        // Probably should do some locking ... somewhere ... maybe here

        typedef typename List::iterator list_iter;
        list_iter end = list.end();
        for (list_iter it = list.begin(); it != end; ++it)
        {
            local_map_[it->first] = it->second;
        }

        return local_map_.size();
    }

    template <typename List>
    List local_map<List>::get(void)
    {
        std::cout << "Getting local set" << std::endl;

        return local_map_;
    }

    // This is a hack: assuming key is always a GID, value a GID
    // This should take two template parameters: Key and Value
    template <typename List>
    naming::id_type local_map<List>::value(naming::id_type key, components::component_type value_type)
    {
        std::cout << "Getting value for key " << key << std::endl;

        // Lock access to the pmap for potential update
        typedef typename mutex_type::scoped_lock scoped_lock;
        scoped_lock l(this);

        naming::id_type value;

        typename List::iterator it = local_map_.find(key);
        if (it == local_map_.end())
        {
            // The key is not in the map
            // Note: this is a very static solution, we are not searching the
            // entire distributed map for the key

            // Create a new instance of value type
            naming::id_type here = hpx::applier::get_applier().get_prefix();
            naming::id_type new_value = components::stubs::runtime_support::create_component(here, value_type, 1);

            // Add it to the map
            local_map_[key] = new_value;

            std::cout << "Key added to map: (" << key << ", " << new_value << ")" << std::endl;

            value = new_value;
        }
        else
        {
            std::cout << "Key already in map: (" << it->first << ", " << it->second << ")" << std::endl;
            // The key was already in the map
            value = it->second;
        }

        util::unlock_the_lock<scoped_lock> ul(l);

        return value;
    }

}}}

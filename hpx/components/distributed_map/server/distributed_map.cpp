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

#include "distributed_map.hpp"
#include "../stubs/distributed_map.hpp"

#include "../local_map.hpp"
#include "../server/local_map.hpp"
#include "../stubs/local_map.hpp"

///////////////////////////////////////////////////////////////////////////////

typedef hpx::components::server::distributed_map<
    std::map<hpx::naming::id_type,hpx::naming::id_type>
> distributed_gids_map_type;

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the distributed_map actions

HPX_REGISTER_ACTION_EX(
    distributed_gids_map_type::get_local_action,
    distributed_gids_map_get_local_action);
HPX_REGISTER_ACTION_EX(
    distributed_gids_map_type::locals_action,
    distributed_gids_map_locals_action);
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<distributed_gids_map_type>,
    distributed_gids_map);
HPX_DEFINE_GET_COMPONENT_TYPE(distributed_gids_map_type);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    template <typename List>
    distributed_map<List>::distributed_map() {}
    
    template <typename List>
    naming::id_type distributed_map<List>::get_local(naming::id_type locale)
    {
        std::cout << "Getting local sublist at " << locale <<  std::endl;

        // If a list already exists on this locality, return a reference to it
        // Otherwise, create a new one and return a reference to it
        if (map_.count(locale) == 0)
        {
            std::cout << "Need to make new sublist" << std::endl;

            // Create a new sub list there
            typedef List map_type;
            typedef hpx::components::local_map<map_type> local_map_type;

            local_map_type props_map(local_map_type::create(locale));

            std::cout << "Created local sublist" << std::endl;

            map_[locale] = props_map.get_gid();
            locals_.push_back(map_[locale]);

            std::cout << "Sending local sublist gid" << std::endl;

            return map_[locale];
        }
        else
        {
            return map_[locale];
        }
    }

    template <typename List>
    std::vector<naming::id_type> distributed_map<List>::locals(void)
    {
        std::cout << "Getting coverage of distributed set" << std::endl;

        return locals_;
    }

}}}

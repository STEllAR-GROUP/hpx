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

#include "local_map.hpp"
#include "../stubs/local_map.hpp"

#define LLOCAL_MAP_(lvl) LAPP_(lvl) << " [LOCAL_MAP] " << gid_ << " "

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
// Must add gids_map_type serialization first
//HPX_REGISTER_ACTION_EX(
//    local_gids_map_type::get_action,
//    local_gids_map_value_action);
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<local_gids_map_type>, local_gids_map);
HPX_DEFINE_GET_COMPONENT_TYPE(local_gids_map_type);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    template <typename List>
    local_map<List>::local_map()
      : gid_(This()->base_type::get_gid()),
        dist_map_(naming::invalid_id),
        local_map_()
    {}

    template <typename List>
    int local_map<List>::append(List list)
    {
        LLOCAL_MAP_(info) << "Appending to local list at locale ";

        lcos::mutex::scoped_lock l(mtx_);

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
        LLOCAL_MAP_(info) << "Getting local set";

        return local_map_;
    }

    // This is a hack: assuming key is always a GID, value a GID
    // This should take two template parameters: Key and Value
    template <typename List>
    naming::id_type local_map<List>::value(naming::id_type key, components::component_type value_type)
    {

        if (local_map_.count(key) == 0)
        {
            // The key was not in the map
            // Note: this is a very static solution, we are not searching the
            // entire distributed map for the key

            // Create a new instance of value type
            naming::id_type value;
            naming::id_type here = hpx::applier::get_applier().get_prefix();
            value = components::stubs::runtime_support::create_component(here, value_type, 1);

            lcos::mutex::scoped_lock l(mtx_);

            if (local_map_.count(key) == 0)
            {
                LLOCAL_MAP_(info) << "Adding key to map: (" << key << ")";

                local_map_[key] = value;

                LLOCAL_MAP_(info) << "Key added to map: (" << key << ", " << value << ")";
            }

        }
        else
        {
            // The key was already in the map
            LLOCAL_MAP_(info) << "Key already in map: (" << key << ", " << local_map_[key] << ")";
        }

        /*
        util::unlock_the_lock<scoped_lock> ul(l);
        */

        return local_map_[key];
    }

}}}

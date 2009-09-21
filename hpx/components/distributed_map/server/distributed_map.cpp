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

#include "distributed_map.hpp"
#include "../stubs/distributed_map.hpp"

#include "../local_map.hpp"
#include "../server/local_map.hpp"
#include "../stubs/local_map.hpp"

#define LDIST_MAP_(lvl) LAPP_(lvl) << " [DIST_MAP] " << gid_ << " "

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
    distributed_map<List>::distributed_map()
      : gid_(This()->base_type::get_gid())
    {}
    
    template <typename List>
    naming::id_type distributed_map<List>::get_local(naming::id_type locale)
    {
        LDIST_MAP_(info) << "distributed_map<>::get_local(" << locale << ")";

        if (map_.count(locale) == 0)
        {
            // Create a new sub list there
            typedef List map_type;
            typedef hpx::components::local_map<map_type> local_map_type;
            local_map_type props_map(local_map_type::create(locale));

            lcos::mutex::scoped_lock l(mtx_);

            if (map_.count(locale) == 0)
            {
                LDIST_MAP_(info) << "Needs to make new sublist";

                //LDIST_MAP_(info) << "Created local sublist " << props_map.get_gid();

                map_[locale] = props_map.get_gid();
                locals_.push_back(map_[locale]);

                LDIST_MAP_(info) << "Sending local sublist gid";
            }

            //mtx_.unlock();
            //return map_[locale];
        }
        else
        {
            LDIST_MAP_(info) << "Returning existing sublist " << map_[locale];
        }

        return map_[locale];
   }

    template <typename List>
    std::vector<naming::id_type> distributed_map<List>::locals(void)
    {
        LDIST_MAP_(info) << "Getting coverage of distributed set";

        return locals_;
    }

}}}

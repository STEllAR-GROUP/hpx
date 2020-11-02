//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/actions_base/plain_action.hpp>
#include <hpx/async_colocated/get_colocation_id.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/components/server/destroy_component.hpp>

#include <hpx/modules/logging.hpp>

#include <cstdint>

HPX_PLAIN_ACTION_ID(hpx::components::server::destroy_component,
    hpx_destroy_component_action, hpx::actions::free_component_action_id);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    void destroy_component(naming::gid_type const& gid,
        naming::address const& addr_)
    {
        naming::address addr(addr_);
        if (addr.locality_ == hpx::get_locality() ||
            agas::is_local_address_cached(gid, addr))
        {
            // Check if component was migrated, we are not interested in
            // pinning the object as it is supposed to be destroyed anyways
            // that is, no one else has a handle to it anymore
            auto r = agas::was_object_migrated(gid,
                [](){ return pinned_ptr(); });

            // The object is local, we can destroy it locally...
            if (!r.first)
            {
                if (naming::refers_to_virtual_memory(gid))
                {
                    // simply delete the memory
                    delete [] reinterpret_cast<std::uint8_t*>(gid.get_lsb());
                    return;
                }

                components::deleter(addr.type_)(gid, addr);

                LRT_(info) << "successfully destroyed component " << gid
                    << " of type: " << components::get_component_type_name(addr.type_);

                return;
            }
        }

        // apply remotely (only if runtime is not stopping)
        naming::id_type id = get_colocation_id(
            launch::sync, naming::id_type(gid, naming::id_type::unmanaged));

        hpx_destroy_component_action()(id, gid, addr);
    }
}}}


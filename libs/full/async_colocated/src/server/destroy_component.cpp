//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/actions_base/plain_action.hpp>
#include <hpx/async_colocated/get_colocation_id.hpp>
#include <hpx/async_colocated/server/destroy_component.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/components_base/detail/agas_interface_functions.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/naming_base/id_type.hpp>

#include <cstdint>

HPX_PLAIN_ACTION_ID(hpx::components::server::destroy_component,
    hpx_destroy_component_action, hpx::actions::free_component_action_id);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server {

    ///////////////////////////////////////////////////////////////////////////
    void destroy_component(
        naming::gid_type const& gid, naming::address const& addr_)
    {
        naming::address addr(addr_);
        if (naming::get_locality_id_from_gid(addr.locality_) ==
                agas::get_locality_id() ||
            agas::is_local_address_cached(gid, addr))
        {
            // Check if component was migrated, we are not interested in
            // pinning the object as it is supposed to be destroyed anyways
            // that is, no one else has a handle to it anymore
            auto r =
                agas::was_object_migrated(gid, []() { return pinned_ptr(); });

            // The object is local, we can destroy it locally...
            if (!r.first)
            {
                if (naming::refers_to_virtual_memory(gid))
                {
                    // simply delete the memory
                    delete[] reinterpret_cast<std::uint8_t*>(gid.get_lsb());
                    return;
                }

                components::deleter(addr.type_)(gid, addr);

                LRT_(info).format(
                    "successfully destroyed component {} of type: {}", gid,
                    components::get_component_type_name(addr.type_));

                return;
            }
        }

        // apply remotely (only if runtime is not stopping)
        naming::id_type id = get_colocation_id(
            launch::sync, naming::id_type(gid, naming::id_type::unmanaged));

        hpx_destroy_component_action()(id, gid, addr);
    }

    ///////////////////////////////////////////////////////////////////////////
    // initialize AGAS interface function pointers in components_base module
    struct HPX_EXPORT destroy_interface_function
    {
        destroy_interface_function()
        {
            agas::detail::destroy_component = &destroy_component;
        }
    };

    destroy_interface_function destroy_init;
}}}    // namespace hpx::components::server

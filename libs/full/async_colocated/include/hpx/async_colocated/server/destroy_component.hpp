//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2011-2017 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/components_base/server/component_heap.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/naming_base/address.hpp>

#include <memory>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server {

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT void destroy_component(
        naming::gid_type const& gid, naming::address const& addr);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    void destroy(naming::gid_type const& gid, naming::address const& addr)
    {
        // make sure this component is located here
        if (agas::get_locality_id() !=
            naming::get_locality_id_from_gid(addr.locality_))
        {
            // This component might have been migrated, find out where it is
            // and instruct that locality to delete it.
            destroy_component(gid, addr);
            return;
        }

        // make sure it's the correct component type
        components::component_type type =
            components::get_component_type<typename Component::wrapped_type>();
        if (!types_are_compatible(type, addr.type_))
        {
            // FIXME: should the component be re-bound ?
            HPX_THROW_EXCEPTION(hpx::error::unknown_component_address,
                "destroy<Component>",
                "global id: {} is not bound to a component "
                "instance of type: {}  (it is bound to a {})",
                gid, get_component_type_name(type),
                get_component_type_name(addr.type_));
            return;
        }

        --instance_count(type);

        // delete the local instances
        Component* c = reinterpret_cast<Component*>(addr.address_);
        c->finalize();
        std::destroy_at(c);
        component_heap<Component>().free(c, 1);
    }

    template <typename Component>
    void destroy(naming::gid_type const& gid)
    {
        naming::address addr;
        if (!agas::resolve_local(gid, addr))
        {
            HPX_THROW_EXCEPTION(hpx::error::unknown_component_address,
                "destroy<Component>",
                "global id: {} is not bound to any component instance", gid);
            return;
        }

        destroy<Component>(gid, addr);
    }
}}}    // namespace hpx::components::server

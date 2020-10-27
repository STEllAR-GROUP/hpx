//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011-2017 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/runtime/components/server/component_heap.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime_fwd.hpp>

#include <sstream>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    HPX_EXPORT void destroy_component(naming::gid_type const& gid,
        naming::address const& addr);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    void destroy(naming::gid_type const& gid, naming::address const& addr)
    {
        // make sure this component is located here
        if (get_locality() != addr.locality_)
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
            std::ostringstream strm;
            strm << "global id " << gid << " is not bound to a component "
                    "instance of type: " << get_component_type_name(type)
                 << " (it is bound to a " << get_component_type_name(addr.type_)
                 << ")";
            HPX_THROW_EXCEPTION(hpx::unknown_component_address,
                "destroy<Component>", strm.str());
            return;
        }

        --instance_count(type);

        // delete the local instances
        Component *c = reinterpret_cast<Component*>(addr.address_);
        c->finalize();
        c->~Component();
        component_heap<Component>().free(c, 1);
    }

    template <typename Component>
    void destroy(naming::gid_type const& gid)
    {

        naming::address addr;
        if (!naming::get_agas_client().resolve_local(gid, addr))
        {
            std::ostringstream strm;
            strm << "global id " << gid << " is not bound to any "
                    "component instance";
            HPX_THROW_EXCEPTION(hpx::unknown_component_address,
                "destroy<Component>", strm.str());
            return;
        }

        destroy<Component>(gid, addr);
    }
}}}



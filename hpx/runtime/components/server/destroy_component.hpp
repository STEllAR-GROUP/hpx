//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_DESTROY_COMPONENT_JUN_02_2008_0146PM)
#define HPX_COMPONENTS_SERVER_DESTROY_COMPONENT_JUN_02_2008_0146PM

#include <hpx/exception.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/util/stringstream.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    void destroy(naming::gid_type const& gid, error_code& ec = throws)
    {
        // retrieve the local address bound to the given global id
        applier::applier& appl = hpx::applier::get_applier();
        naming::address addr;
        if (!appl.get_agas_client().resolve(gid, addr))
        {
            hpx::util::osstream strm;
            strm << "global id " << gid << " is not bound to any "
                    "component instance";
            HPX_THROWS_IF(ec, hpx::unknown_component_address,
                "destroy<Component>", hpx::util::osstream_get_string(strm));
            return;
        }

        // make sure this component is located here
        if (appl.here() != addr.locality_)
        {
            // FIXME: should the component be re-bound ?
            hpx::util::osstream strm;
            strm << "global id " << gid << " is not bound to any local "
                    "component instance";
            HPX_THROWS_IF(ec, hpx::unknown_component_address,
                "destroy<Component>", hpx::util::osstream_get_string(strm));
            return;
        }

        // make sure it's the correct component type
        components::component_type type =
            components::get_component_type<typename Component::wrapped_type>();
        if (!types_are_compatible(type, addr.type_))
        {
            // FIXME: should the component be re-bound ?
            hpx::util::osstream strm;
            strm << "global id " << gid << " is not bound to a component "
                    "instance of type: " << get_component_type_name(type)
                 << " (it is bound to a " << get_component_type_name(addr.type_)
                 << ")";
            HPX_THROWS_IF(ec, hpx::unknown_component_address,
                "destroy<Component>", hpx::util::osstream_get_string(strm));
            return;
        }

        // delete the local instances
        Component::destroy(reinterpret_cast<Component*>(addr.address_));
        if (&ec != &throws)
            ec = make_success_code();
    }
}}}

#endif


//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_DESTROY_COMPONENT_JUN_02_2008_0146PM)
#define HPX_COMPONENTS_SERVER_DESTROY_COMPONENT_JUN_02_2008_0146PM

#include <hpx/config.hpp>
#include <hpx/error_code.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/one_size_heap_list_base.hpp>

#include <sstream>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    HPX_EXPORT void destroy_component(naming::gid_type const& gid,
        naming::address const& addr, error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    void destroy(naming::gid_type const& gid, naming::address const& addr,
        error_code& ec = throws)
    {
        // make sure this component is located here
        if (get_locality() != addr.locality_)
        {
            // This component might have been migrated, find out where it is
            // and instruct that locality to delete it.
            destroy_component(gid, addr, ec);
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
            HPX_THROWS_IF(ec, hpx::unknown_component_address,
                "destroy<Component>", strm.str());
            return;
        }

        // delete the local instances
        Component::destroy(reinterpret_cast<Component*>(addr.address_));
        if (&ec != &throws)
            ec = make_success_code();
    }

    template <typename Component>
    void destroy(naming::gid_type const& gid, error_code& ec = throws)
    {
        // retrieve the local address bound to the given global id
        applier::applier& appl = hpx::applier::get_applier();

        naming::address addr;
        if (!appl.get_agas_client().resolve_local(gid, addr))
        {
            std::ostringstream strm;
            strm << "global id " << gid << " is not bound to any "
                    "component instance";
            HPX_THROWS_IF(ec, hpx::unknown_component_address,
                "destroy<Component>", strm.str());
            return;
        }

        destroy<Component>(gid, addr, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT void destroy_base_lco(naming::gid_type const& gid,
        util::one_size_heap_list_base* heap, components::component_type type,
        error_code& ec = throws);

    HPX_EXPORT void destroy_base_lco(naming::gid_type const& gid,
        naming::address const& addr, util::one_size_heap_list_base* heap,
        components::component_type type, error_code& ec = throws);
}}}

#endif


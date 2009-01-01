//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_MANAGE_COMPONENT_JUN_02_2008_0146PM)
#define HPX_COMPONENTS_SERVER_MANAGE_COMPONENT_JUN_02_2008_0146PM

#include <hpx/exception.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/util/util.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    naming::id_type create (std::size_t count)
    {
        if (0 == count) {
            HPX_THROW_EXCEPTION(hpx::bad_parameter, "count shouldn't be zero");
            return naming::invalid_id;
        }

        Component* c = static_cast<Component*>(Component::create(count));
        naming::id_type gid = c->get_gid();
        if (gid) 
            return gid;

        delete c;
        HPX_THROW_EXCEPTION(hpx::duplicate_component_address,
            "global id is already bound to a different "
            "component instance");
        return naming::invalid_id;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    void destroy(naming::id_type const& gid)
    {
        // retrieve the local address bound to the given global id
        applier::applier& appl = hpx::applier::get_applier();
        naming::address addr;
        if (!appl.get_agas_client().resolve(gid, addr)) 
        {
            HPX_THROW_EXCEPTION(hpx::unknown_component_address,
                "global id is not bound to any component instance");
        }

        // make sure this component is located here
        if (appl.here() != addr.locality_) 
        {
            // FIXME: should the component be re-bound ?
            HPX_THROW_EXCEPTION(hpx::unknown_component_address,
                "global id is not bound to any local component instance");
        }

        // make sure it's the correct component type
        components::component_type type = 
            components::get_component_type<typename Component::wrapped_type>();
        if (!types_are_compatible(type, addr.type_))
        {
            // FIXME: should the component be re-bound ?
            HPX_OSSTREAM strm;
            strm << "global id is not bound to a component instance of type: "
                 << get_component_type_name(type);
            HPX_THROW_EXCEPTION(hpx::unknown_component_address,
                HPX_OSSTREAM_GETSTRING(strm));
        }

        // delete the local instances
        Component::destroy(reinterpret_cast<Component*>(addr.address_));
    }

}}}

#endif

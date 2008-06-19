//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_MANAGE_COMPONENT_JUN_02_2008_0146PM)
#define HPX_COMPONENTS_SERVER_MANAGE_COMPONENT_JUN_02_2008_0146PM

#include <boost/throw_exception.hpp>

#include <hpx/exception.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/applier/applier.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    naming::id_type create (applier::applier& appl)
    {
        Component* c = new Component;
        naming::id_type gid = c->get_gid();
        if (!appl.get_dgas_client().bind(gid, 
                naming::address(appl.here(), Component::value, c))) 
        {
            delete c;
            boost::throw_exception(
                hpx::exception(hpx::duplicate_component_address,
                    "global id is already bound to a different "
                    "component instance"));
            return naming::invalid_id;
        }
        return gid;
    }
    
    template <typename Component, typename Arg0>
    naming::id_type create (applier::applier& appl, Arg0 const& arg0)
    {
        Component* c = new Component(arg0);
        naming::id_type gid = c->get_gid();
        if (!appl.get_dgas_client().bind(gid, 
                naming::address(appl.here(), Component::value, c))) 
        {
            delete c;
            boost::throw_exception(
                hpx::exception(hpx::duplicate_component_address,
                    "global id is already bound to a different "
                    "component instance"));
            return naming::invalid_id;
        }
        return gid;
    }

    // bring in overload for 2 and more arguments
    #include <hpx/components/server/manage_component_implementations.hpp>

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    void destroy(applier::applier& appl, naming::id_type const& gid)
    {
        // retrieve the local address bound to the given global id
        naming::address addr;
        if (!appl.get_dgas_client().unbind(gid, addr)) 
        {
            boost::throw_exception(
                hpx::exception(hpx::unknown_component_address,
                    "global id is not bound to any component instance"));
        }
        
        // make sure this component is located here
        if (appl.here() != addr.locality_) 
        {
            // FIXME: should the component be re-bound ?
            boost::throw_exception(
                hpx::exception(hpx::unknown_component_address,
                    "global id is not bound to any local component instance"));
        }
        
        // make sure it's the correct component type
        if (Component::value != addr.type_)
        {
            // FIXME: should the component be re-bound ?
            boost::throw_exception(
                hpx::exception(hpx::unknown_component_address,
                    std::string("global id is not bound to a component instance of type") +
                    get_component_type_name(Component::value)));
        }
        
        // delete the local instance
        delete reinterpret_cast<Component*>(addr.address_);
    }
    
}}}

#endif

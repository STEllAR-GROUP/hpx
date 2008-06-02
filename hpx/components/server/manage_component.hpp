//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_MANAGE_COMPONENT_JUN_02_2008_0146PM)
#define HPX_COMPONENTS_SERVER_MANAGE_COMPONENT_JUN_02_2008_0146PM

namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    Component* create (naming::resolver_client const& dgas, naming::id_type gid)
    {
        Component* c = new Component;
        if (!dgas.bind(gid, naming::address(dgas.here(), Component::value, c))) 
        {
            delete c;
            boost::throw_exception(
                hpx::exception(hpx::duplicate_component_address,
                    "global id is already bound to a different "
                    "component instance"));
        }
        return c;
    }
    
    template <typename Component, typename Arg0>
    Component* create (naming::resolver_client const& dgas, naming::id_type gid,
        Arg0 const& arg0)
    {
        Component* c = new Component(arg0);
        if (!dgas.bind(gid, naming::address(dgas.here(), Component::value, c))) 
        {
            delete c;
            boost::throw_exception(
                hpx::exception(hpx::duplicate_component_address,
                    "global id is already bound to a different "
                    "component instance"));
        }
        return c;
    }
    
    template <typename Component, typename Arg0, typename Arg1>
    Component* create (naming::resolver_client const& dgas, naming::id_type gid,
        Arg0 const& arg0, Arg1 const& arg1)
    {
        Component* c = new Component(arg0, arg1);
        if (!dgas.bind(gid, naming::address(dgas.here(), Component::value, c))) 
        {
            delete c;
            boost::throw_exception(
                hpx::exception(hpx::duplicate_component_address,
                    "global id is already bound to a different "
                    "component instance"));
        }
        return c;
    }

    // bring in overload for more than 2 arguments
    #include <hpx/components/server/manage_component_implementations.hpp>

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    void destroy(naming::resolver_client& dgas, naming::id_type gid)
    {
        // retrieve the local address bound to the given global id
        naming::address addr;
        if (!dgas.unbind(gid, addr)) 
        {
            boost::throw_exception(
                hpx::exception(hpx::unknown_component_address,
                    "global id is not bound to any component instance"));
        }
        
        // make sure this component is located here
        if (dgas.here() != addr.locality_) 
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

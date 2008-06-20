//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_FACTORY_JUN_02_2008_1145AM)
#define HPX_COMPONENTS_FACTORY_JUN_02_2008_1145AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/components/component_type.hpp>
#include <hpx/components/action.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>

namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class factory
    {
    public:
        // parcel action code: the action to be performed on the destination 
        // object (the accumulator)
        enum actions
        {
            factory_create = 0,  // create a new component, no arguments
            factory_free = 1,    // delete an existing component, no arguments
        };

        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        enum { value = component_factory };

        // constructor
        factory()
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// \brief Action to create new components
        threadmanager::thread_state create(
            threadmanager::px_thread_self& self, applier::applier& app,
            naming::id_type* gid, components::component_type type, 
            std::size_t count); 

        /// \brief Action to delete existing components
        threadmanager::thread_state free(
            threadmanager::px_thread_self& self, applier::applier& app,
            components::component_type type, naming::id_type const& gid,
            std::size_t count); 

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into a action
        // type, allowing to generate all require boilerplate code for threads,
        // serialization, etc.
        typedef result_action2<
            factory, naming::id_type, factory_create, 
            components::component_type, std::size_t, &factory::create
        > create_action;

        typedef action3<
            factory, factory_free, components::component_type, 
            naming::id_type const&, std::size_t, &factory::free
        > free_action;
    };

}}}

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the factory actions
HPX_SERIALIZE_ACTION(hpx::components::server::factory::create_action);
HPX_SERIALIZE_ACTION(hpx::components::server::factory::free_action);

#endif

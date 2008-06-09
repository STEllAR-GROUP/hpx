//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_FACTORY_JUN_02_2008_1145AM)
#define HPX_COMPONENTS_FACTORY_JUN_02_2008_1145AM

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <boost/serialization/export.hpp>

#include <hpx/components/component_type.hpp>
#include <hpx/components/action.hpp>
#include <hpx/naming/resolver_client.hpp>

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
            create_component = 0,    // create a new component, no arguments
        };

        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        enum { value = component_factory };
        
        // constructor
        factory(naming::resolver_client const& dgas)
          : dgas_(dgas)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        
        /// create a new component
        threadmanager::thread_state create(
            threadmanager::px_thread_self& self, applier::applier& app,
            components::component_type type, naming::id_type gid); 
        
        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into a action
        // type, allowing to generate all require boilerplate code for threads,
        // serialization, etc.
        typedef action2<
            factory, create_component, 
            components::component_type, naming::id_type, &factory::create
        > create_action;
        
    private:
        naming::resolver_client const& dgas_;
    };
    
}}}

///////////////////////////////////////////////////////////////////////////////
// enable serialization support (these need to be in the global namespace)
BOOST_CLASS_EXPORT(hpx::components::server::factory::create_action);

#endif

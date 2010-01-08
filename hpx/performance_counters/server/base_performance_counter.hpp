//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_SERVER_BASE_MAR_03_2009_0741M)
#define HPX_PERFORMANCE_COUNTERS_SERVER_BASE_MAR_03_2009_0741M

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/performance_counters/counters.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    // parcel action code: the action to be performed on the destination 
    // object 
    enum actions
    {
        performance_counter_get_counter_info = 0,
        performance_counter_get_counter_value = 1,
    };

    class base_performance_counter 
    {
    protected:
        /// Destructor, needs to be virtual to allow for clean destruction of
        /// derived objects
        virtual ~base_performance_counter() {}

        virtual void get_counter_info(counter_info& info) = 0;
        virtual void get_counter_value(counter_value& value) = 0;

    public:
        // components must contain a typedef for wrapping_type defining the
        // simple_component type used to encapsulate instances of this 
        // component
        typedef components::managed_component<base_performance_counter> wrapping_type;

        /// \brief finalize() will be called just before the instance gets 
        ///        destructed
        void finalize() {}

        // This is the component id. Every component needs to have a function
        // \a get_component_type() which is used by the generic action 
        // implementation to associate this component with a given action.
        static components::component_type get_component_type() 
        { 
            return components::component_performance_counter; 
        }
        static void set_component_type(components::component_type) 
        { 
        }

        ///////////////////////////////////////////////////////////////////////
        counter_info get_counter_info_nonvirt()
        {
            counter_info info;
            get_counter_info(info);
            return info;
        }

        counter_value get_counter_value_nonvirt()
        {
            counter_value value;
            get_counter_value(value);
            return value;
        }

        /// Each of the exposed functions needs to be encapsulated into an action
        /// type, allowing to generate all required boilerplate code for threads,
        /// serialization, etc.

        /// The \a get_counter_info_action may be used to ...
        typedef hpx::actions::result_action0<
            base_performance_counter, counter_info, 
            performance_counter_get_counter_info, 
            &base_performance_counter::get_counter_info_nonvirt
        > get_counter_info_action;

        /// The \a get_counter_value_action may be used to ...
        typedef hpx::actions::result_action0<
            base_performance_counter, counter_value, 
            performance_counter_get_counter_value, 
            &base_performance_counter::get_counter_value_nonvirt
        > get_counter_value_action;
    };

}}}

#endif


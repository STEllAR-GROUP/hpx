//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_COMPONENTS_SERVER_PLAIN_COMPONENT_NOV_14_2008_0726PM)
#define HPX_RUNTIME_COMPONENTS_SERVER_PLAIN_COMPONENT_NOV_14_2008_0726PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>

#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    /// placeholder type allowing to integrate the plain action templates
    /// with the existing component based action template infrastructure
    template <typename Action>
    struct plain_function
    {
        static component_type get_component_type()
        {
            return components::get_component_type<plain_function<Action> >();
        }
        static void set_component_type(component_type type)
        {
            components::set_component_type<plain_function<Action> >(type);
        }

        /// This is the default hook implementation for decorate_action which 
        /// does no hooking at all.
        template <typename F>
        static threads::thread_function_type
        decorate_action(naming::address::address_type, F && f)
        {
            return std::forward<F>(f);
        }

        /// This is the default hook implementation for schedule_thread which
        /// forwards to the default scheduler.
        static void schedule_thread(naming::address::address_type,
            threads::thread_init_data& data,
            threads::thread_state_enum initial_state)
        {
            hpx::threads::register_work_plain(data, initial_state); //-V106
        }
    };
}}}

#endif



//  Copyright (c) 2007-2011 Hartmut Kaiser
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

        virtual void get_counter_value(counter_value& value) = 0;

    public:
        base_performance_counter() {}
        base_performance_counter(counter_info const& info)
          : info_(info)
        {}

        // components must contain a typedef for wrapping_type defining the
        // simple_component type used to encapsulate instances of this
        // component
        typedef components::managed_component<base_performance_counter> wrapping_type;
        typedef base_performance_counter base_type_holder;

        /// \brief finalize() will be called just before the instance gets
        ///        destructed
        void finalize() {}

        static components::component_type get_component_type()
        {
            return components::get_component_type<wrapping_type>();
        }
        static void set_component_type(components::component_type t)
        {
            components::set_component_type<wrapping_type>(t);
        }

        ///////////////////////////////////////////////////////////////////////
        counter_info get_counter_info_nonvirt()
        {
            return info_;
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

        /// The \a get_counter_info_action retrieves a performance counters
        /// information.
        typedef hpx::actions::result_action0<
            base_performance_counter, counter_info,
            performance_counter_get_counter_info,
            &base_performance_counter::get_counter_info_nonvirt,
            threads::thread_priority_critical
        > get_counter_info_action;

        /// The \a get_counter_value_action queries the value of a performance
        /// counter.
        typedef hpx::actions::result_action0<
            base_performance_counter, counter_value,
            performance_counter_get_counter_value,
            &base_performance_counter::get_counter_value_nonvirt,
            threads::thread_priority_critical
        > get_counter_value_action;

    protected:
        hpx::performance_counters::counter_info info_;
    };

}}}

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::performance_counters::server::base_performance_counter::get_counter_info_action,
    performance_counter_get_counter_info_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::performance_counters::server::base_performance_counter::get_counter_value_action,
    performance_counter_get_counter_value_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<hpx::performance_counters::counter_info>::set_result_action,
    set_result_action_counter_info);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<hpx::performance_counters::counter_value>::set_result_action,
    set_result_action_counter_value);

#endif


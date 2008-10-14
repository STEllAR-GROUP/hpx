//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_BASE_LCO_JUN_12_2008_0852PM)
#define HPX_LCOS_BASE_LCO_JUN_12_2008_0852PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/action.hpp>

namespace hpx { namespace lcos 
{
    // parcel action code: the action to be performed on the destination 
    // object 
    enum actions
    {
        lco_set_event = 0,
        lco_set_result = 1,
        lco_set_error = 2
    };

    /// The \a base_lco class is the common base class for all LCO's 
    /// implementing a simple set_event action
    struct base_lco
    {
        // components must contain a typedef for wrapping_type defining the
        // managed_component_base type used to encapsulate instances of this 
        // component
        typedef components::managed_component_base<base_lco> wrapping_type;

        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        static components::component_type get_component_type() 
        { 
            return components::component_base_lco; 
        }

        /// Destructor, needs to be virtual to allow for clean destruction of
        /// derived objects
        virtual ~base_lco() {}

        virtual threads::thread_state set_event (
            threads::thread_self& self, applier::applier& appl) = 0;

        ///
        virtual threads::thread_state set_error (
            threads::thread_self& self, applier::applier& appl,
            hpx::error code, std::string msg)
        {
            // just rethrow the exception
            HPX_THROW_EXCEPTION(code, msg);
            return threads::terminated;
        }

        /// actions
        threads::thread_state set_event_nonvirt (
            threads::thread_self& self, applier::applier& appl)
        {
            return set_event(self, appl);
        }

        threads::thread_state set_error_nonvirt (
            threads::thread_self& self, applier::applier& appl,
            hpx::error code, std::string msg)
        {
            return set_error(self, appl, code, msg);
        }

        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::action0<
            base_lco, lco_set_event, &base_lco::set_event_nonvirt
        > set_event_action;

        typedef hpx::actions::action2<
            base_lco, lco_set_error, hpx::error, std::string,
            &base_lco::set_error_nonvirt
        > set_error_action;
    };

    /// The \a base_lco_with_value class is the common base class for all LCO's 
    /// synchronizing on a value. 
    /// The \a Result template argument should be set to the type of the 
    /// argument expected for the set_result action.
    template <typename Result>
    struct base_lco_with_value : base_lco
    {
        // components must contain a typedef for wrapping_type defining the
        // managed_component_base type used to encapsulate instances of this 
        // component
        typedef 
            components::managed_component_base<base_lco_with_value> 
        wrapping_type;

        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        static components::component_type get_component_type() 
        { 
            return components::component_base_lco_with_value; 
        }

        /// Destructor, needs to be virtual to allow for clean destruction of
        /// derived objects
        virtual ~base_lco_with_value() {}

        virtual threads::thread_state set_event (
            threads::thread_self& self, applier::applier& appl)
        {
            return set_result(self, appl, Result());
        }

        ///
        virtual threads::thread_state set_result (
            threads::thread_self&, applier::applier& appl,
            Result const& result) = 0;

        /// actions
        threads::thread_state set_result_nonvirt (
            threads::thread_self& self, applier::applier& appl,
            Result const& result) 
        {
            return set_result(self, appl, result);
        }

        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::action1<
            base_lco_with_value, lco_set_result, Result const&, 
            &base_lco_with_value::set_result_nonvirt
        > set_result_action;
    };

    /// The base_lco<void> specialization is used whenever the set_event action
    /// for a particular LCO doesn't carry an argument
    template<>
    struct base_lco_with_value<void> : public base_lco
    {
        // components must contain a typedef for wrapping_type defining the
        // managed_component_base type used to encapsulate instances of this 
        // component
        typedef 
            components::managed_component_base<base_lco_with_value> 
        wrapping_type;

        /// Destructor, needs to be virtual to allow for clean destruction of
        /// derived objects
        virtual ~base_lco_with_value() {}
    };

}}

#endif

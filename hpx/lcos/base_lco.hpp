//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_BASE_LCO_JUN_12_2008_0852PM)
#define HPX_LCOS_BASE_LCO_JUN_12_2008_0852PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/components/component_type.hpp>
#include <hpx/components/action.hpp>

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
        // wrapper type used to encapsulate instances of this component
        typedef components::wrapper<base_lco> wrapping_type;

        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        enum { value = components::component_base_lco };

        /// Destructor, needs to be virtual to allow for clean destruction of
        /// derived objects
        virtual ~base_lco() {}

        ///
        virtual threadmanager::thread_state set_event (
            threadmanager::px_thread_self&, applier::applier& appl) = 0;

        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef components::action0<
            base_lco, lco_set_event, &base_lco::set_event
        > set_event_action;
    };

    /// The \a base_lco_with_value class is the common base class for all LCO's 
    /// synchronizing on a value. 
    /// The \a Result template argument should be set to the type of the 
    /// argument expected for the set_result action.
    template <typename Result>
    struct base_lco_with_value
    {
        // components must contain a typedef for wrapping_type defining the
        // wrapper type used to encapsulate instances of this component
        typedef components::wrapper<base_lco_with_value> wrapping_type;

        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        enum { value = components::component_base_lco_with_value };

        /// Destructor, needs to be virtual to allow for clean destruction of
        /// derived objects
        virtual ~base_lco_with_value() {}

        ///
        virtual threadmanager::thread_state set_result (
            threadmanager::px_thread_self&, applier::applier& appl,
            Result const& result) = 0;

        ///
        virtual threadmanager::thread_state set_error (
            threadmanager::px_thread_self&, applier::applier& appl,
            hpx::error code, std::string msg) = 0;

        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef components::action1<
            base_lco_with_value, lco_set_result, Result const&, 
            &base_lco_with_value::set_result
        > set_result_action;

        typedef components::action2<
            base_lco_with_value, lco_set_error, hpx::error, std::string,
            &base_lco_with_value::set_error
        > set_error_action;
    };

    /// The base_lco<void> specialization is used whenever the set_event action
    /// for a particular LCO doesn't carry an argument
    template<>
    struct base_lco_with_value<void> : public base_lco
    {
        // components must contain a typedef for wrapping_type defining the
        // wrapper type used to encapsulate instances of this component
        typedef components::wrapper<base_lco_with_value> wrapping_type;

        /// Destructor, needs to be virtual to allow for clean destruction of
        /// derived objects
        virtual ~base_lco_with_value() {}

        ///
        virtual threadmanager::thread_state set_event (
            threadmanager::px_thread_self&, applier::applier&) = 0;

        ///
        virtual threadmanager::thread_state set_error (
            threadmanager::px_thread_self&, applier::applier&,
            hpx::error, std::string) = 0;

        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef components::action0<
            base_lco_with_value, lco_set_result, &base_lco_with_value::set_event
        > set_result_action;

        typedef components::action2<
            base_lco_with_value, lco_set_error, hpx::error, std::string,
            &base_lco_with_value::set_error
        > set_error_action;
    };

}}

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the future actions
HPX_SERIALIZE_ACTION(hpx::lcos::base_lco::set_event_action);

HPX_SERIALIZE_ACTION(hpx::lcos::base_lco_with_value<hpx::naming::id_type>::set_result_action);
HPX_SERIALIZE_ACTION(hpx::lcos::base_lco_with_value<hpx::naming::id_type>::set_error_action);
HPX_SERIALIZE_ACTION(hpx::lcos::base_lco_with_value<double>::set_result_action);
HPX_SERIALIZE_ACTION(hpx::lcos::base_lco_with_value<double>::set_error_action);

#endif

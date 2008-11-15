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
#include <hpx/runtime/actions/component_action.hpp>

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

    /// \class base_lco base_lco.hpp hpx/lcos/base_lco.hpp
    /// 
    /// The \a base_lco class is the common base class for all LCO's 
    /// implementing a simple set_event action
    class base_lco 
    {
    protected:
        virtual threads::thread_state set_event (
            threads::thread_self& self, applier::applier& appl) = 0;

        virtual threads::thread_state set_error (
            threads::thread_self& self, applier::applier& appl,
            hpx::error code, std::string const& msg)
        {
            // just rethrow the exception
            HPX_THROW_EXCEPTION(code, msg);
            return threads::terminated;
        }

    public:
        // components must contain a typedef for wrapping_type defining the
        // managed_component type used to encapsulate instances of this 
        // component
        typedef components::managed_component<base_lco> wrapping_type;

        // This is the component id. Every component needs to have a function
        // \a get_component_type() which is used by the generic action 
        // implementation to associate this component with a given action.
        static components::component_type get_component_type() 
        { 
            return components::component_base_lco; 
        }
        static void set_component_type(components::component_type) 
        { 
        }

        /// Destructor, needs to be virtual to allow for clean destruction of
        /// derived objects
        virtual ~base_lco() {}

        /// \brief finalize() will be called just before the instance gets 
        ///        destructed
        ///
        /// \param self [in] The PX \a thread used to execute this function.
        /// \param appl [in] The applier to be used for finalization of the 
        ///             component instance. 
        void finalize(threads::thread_self& self, applier::applier& appl) {}

        /// The \a function set_event_nonvirt is called whenever a 
        /// \a set_event_action is applied on a instance of a LCO. This function
        /// just forwards to the virtual function \a set_event, which is 
        /// overloaded by the derived concrete LCO.
        ///
        /// \param self   [in] The PX \a thread used to execute this function.
        /// \param appl   [in] The \a applier instance to be used to execute 
        ///               the embedded action.
        ///
        /// \returns      The thread state the calling thread needs to be set
        ///               to after returning from this function.
        threads::thread_state set_event_nonvirt (
            threads::thread_self& self, applier::applier& appl)
        {
            return set_event(self, appl);
        }

        /// The \a function set_error_nonvirt is called whenever a 
        /// \a set_error_action is applied on a instance of a LCO. This function
        /// just forwards to the virtual function \a set_error, which is 
        /// overloaded by the derived concrete LCO.
        ///
        /// \param self   [in] The PX \a thread used to execute this function.
        /// \param appl   [in] The \a applier instance to be used to execute 
        ///               the embedded action.
        /// \param code   [in] The error code of the error to report to this 
        ///               LCO instance.
        /// \param msg    [in] The error message describing the error to report
        ///               to this LCO instance.
        ///
        /// \returns      The thread state the calling thread needs to be set
        ///               to after returning from this function.
        threads::thread_state set_error_nonvirt (
            threads::thread_self& self, applier::applier& appl,
            hpx::error code, std::string const& msg)
        {
            return set_error(self, appl, code, msg);
        }

        /// Each of the exposed functions needs to be encapsulated into an action
        /// type, allowing to generate all required boilerplate code for threads,
        /// serialization, etc.
        ///
        /// The \a set_event_action may be used to unconditionally trigger any 
        /// LCO instances, it carries no additional parameters.
        typedef hpx::actions::action0<
            base_lco, lco_set_event, &base_lco::set_event_nonvirt
        > set_event_action;

        /// The \a set_error_action may be used to transfer arbitrary error 
        /// information from the remote site to the LCO instance specified as
        /// a continuation. This action carries 2 parameters:
        ///
        /// \param hpx::error   [in] The type of the error code of an error 
        ///                     to report to this LCO instance.
        /// \param std::string  [in] The type of the error message describing 
        ///                     an error to report to this LCO instance.
        typedef hpx::actions::action2<
            base_lco, lco_set_error, hpx::error, std::string const&,
            &base_lco::set_error_nonvirt
        > set_error_action;
    };

    /// \class base_lco_with_value base_lco.hpp hpx/lcos/base_lco.hpp
    /// 
    /// The \a base_lco_with_value class is the common base class for all LCO's 
    /// synchronizing on a value. 
    /// The \a Result template argument should be set to the type of the 
    /// argument expected for the set_result action.
    ///
    /// \tparam Result The type of the result value to be carried back to the 
    ///                LCO instance.
    template <typename Result>
    class base_lco_with_value : public base_lco
    {
    protected:
        /// Destructor, needs to be virtual to allow for clean destruction of
        /// derived objects
        virtual ~base_lco_with_value() {}

        virtual threads::thread_state set_event (
            threads::thread_self& self, applier::applier& appl)
        {
            return set_result(self, appl, Result());
        }

        virtual threads::thread_state set_result (
            threads::thread_self&, applier::applier& appl,
            Result const& result) = 0;

    public:
        // components must contain a typedef for wrapping_type defining the
        // managed_component type used to encapsulate instances of this 
        // component
        typedef components::managed_component<base_lco_with_value> wrapping_type;

        // This is the component id. Every component needs to have a function
        // \a get_component_type() which is used by the generic action 
        // implementation to associate this component with a given action.
        static components::component_type get_component_type() 
        { 
            return components::component_base_lco_with_value; 
        }

        /// The \a function set_result_nonvirt is called whenever a 
        /// \a set_result_action is applied on a instance of a LCO. This 
        /// function just forwards to the virtual function \a set_result, which 
        /// is overloaded by the derived concrete LCO.
        ///
        /// \param self   [in] The PX \a thread used to execute this function.
        /// \param appl   [in] The \a applier instance to be used to execute 
        ///               the embedded action.
        /// \param result [in] The result value to be transferred from the 
        ///               remote operation back to this LCO instance.
        ///
        /// \returns      The thread state the calling thread needs to be set
        ///               to after returning from this function.
        threads::thread_state set_result_nonvirt (
            threads::thread_self& self, applier::applier& appl,
            Result const& result) 
        {
            return set_result(self, appl, result);
        }

        /// Each of the exposed functions needs to be encapsulated into an action
        /// type, allowing to generate all required boilerplate code for threads,
        /// serialization, etc.
        ///
        /// The \a set_result_action may be used to trigger any LCO instances
        /// while carrying an additional parameter of any type.
        ///
        /// \param Result [in] The type of the result to be transferred back to 
        ///               this LCO instance.
        typedef hpx::actions::action1<
            base_lco_with_value, lco_set_result, Result const&, 
            &base_lco_with_value::set_result_nonvirt
        > set_result_action;
    };

    /// \class base_lco_with_value base_lco.hpp hpx/lcos/base_lco.hpp
    /// 
    /// The base_lco<void> specialization is used whenever the set_event action
    /// for a particular LCO doesn't carry any argument. 
    ///
    /// \tparam void This specialization expects no result value and is almost
    ///              completely equivalent to the plain \a base_lco.
    template <>
    class base_lco_with_value<void> : public base_lco
    {
    protected:
        /// Destructor, needs to be virtual to allow for clean destruction of
        /// derived objects
        virtual ~base_lco_with_value() {}

    public:
        // components must contain a typedef for wrapping_type defining the
        // managed_component type used to encapsulate instances of this 
        // component
        typedef components::managed_component<base_lco_with_value> wrapping_type;
    };

}}

#endif

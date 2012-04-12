//  Copyright (c) 2007-2012 Hartmut Kaiser
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
#include <hpx/util/ini.hpp>
#include <type_traits>

#include <boost/mpl/bool.hpp>

namespace hpx { namespace lcos
{
    /// The \a base_lco class is the common base class for all LCO's
    /// implementing a simple set_event action
    class base_lco
    {
    protected:
        virtual void set_event () = 0;

        virtual void set_exception (boost::exception_ptr const& e)
        {
            // just rethrow the exception
            boost::rethrow_exception(e);
        }

        // noop by default
        virtual void connect(naming::id_type const &)
        {
        }

        // noop by default
        virtual void disconnect(naming::id_type const &)
        {
        }

    public:
        // components must contain a typedef for wrapping_type defining the
        // managed_component type used to encapsulate instances of this
        // component
        typedef components::managed_component<base_lco> wrapping_type;
        typedef base_lco base_type_holder;

        static components::component_type get_component_type()
        {
            return components::get_component_type<base_lco>();
        }
        static void set_component_type(components::component_type type)
        {
            components::set_component_type<base_lco>(type);
        }

        /// Destructor, needs to be virtual to allow for clean destruction of
        /// derived objects
        virtual ~base_lco() {}

        /// \brief finalize() will be called just before the instance gets
        ///        destructed
        void finalize() {}

        /// The \a function set_event_nonvirt is called whenever a
        /// \a set_event_action is applied on a instance of a LCO. This function
        /// just forwards to the virtual function \a set_event, which is
        /// overloaded by the derived concrete LCO.
        void set_event_nonvirt()
        {
            set_event();
        }

        /// The \a function set_exception is called whenever a
        /// \a set_error_action is applied on a instance of a LCO. This function
        /// just forwards to the virtual function \a set_exception, which is
        /// overloaded by the derived concrete LCO.
        ///
        /// \param e      [in] The exception encapsulating the error to report
        ///               to this LCO instance.
        void set_exception_nonvirt (boost::exception_ptr const& e)
        {
            set_exception(e);
        }

        /// The \a function connect_nonvirt is called whenever a
        /// \a connect_action is applied on a instance of a LCO. This function
        /// just forwards to the virtual function \a connect, which is
        /// overloaded by the derived concrete LCO.
        ///
        /// \param id [in] target id
        void connect_nonvirt(naming::id_type const & id)
        {
            connect(id);
        }

        /// The \a function disconnect_nonvirt is called whenever a
        /// \a disconnect_action is applied on a instance of a LCO. This function
        /// just forwards to the virtual function \a disconnect, which is
        /// overloaded by the derived concrete LCO.
        ///
        /// \param id [in] target id
        void disconnect_nonvirt(naming::id_type const & id)
        {
            disconnect(id);
        }

    public:
        /// Each of the exposed functions needs to be encapsulated into an action
        /// type, allowing to generate all required boilerplate code for threads,
        /// serialization, etc.
        ///
        /// The \a set_event_action may be used to unconditionally trigger any
        /// LCO instances, it carries no additional parameters.
        HPX_COMPONENT_DIRECT_ACTION(base_lco, set_event_nonvirt,
            set_event_action);

        /// The \a set_exception_action may be used to transfer arbitrary error
        /// information from the remote site to the LCO instance specified as
        /// a continuation. This action carries 2 parameters:
        ///
        /// \param boost::exception_ptr
        ///               [in] The exception encapsulating the error to report
        ///               to this LCO instance.
        HPX_COMPONENT_DIRECT_ACTION(base_lco, set_exception_nonvirt,
            set_exception_action);

        /// The \a connect_action may be used to
        HPX_COMPONENT_DIRECT_ACTION(base_lco, connect_nonvirt, connect_action);

        /// The \a set_exception_action may be used to
        HPX_COMPONENT_DIRECT_ACTION(base_lco, disconnect_nonvirt,
            disconnect_action);
    };

    /// The \a base_lco_with_value class is the common base class for all LCO's
    /// synchronizing on a value.
    /// The \a RemoteResult template argument should be set to the type of the
    /// argument expected for the set_value action.
    ///
    /// \tparam RemoteResult The type of the result value to be carried back
    /// to the LCO instance.
    template <typename Result, typename RemoteResult>
    class base_lco_with_value : public base_lco
    {
    protected:
        /// Destructor, needs to be virtual to allow for clean destruction of
        /// derived objects
        virtual ~base_lco_with_value() {}

        virtual void set_event()
        {
            set_value(RemoteResult());
        }

        virtual void set_value (BOOST_RV_REF(RemoteResult) result) = 0;

        virtual Result get_value() = 0;

    public:
        // components must contain a typedef for wrapping_type defining the
        // managed_component type used to encapsulate instances of this
        // component
        typedef components::managed_component<base_lco_with_value> wrapping_type;
        typedef base_lco_with_value base_type_holder;

        static components::component_type get_component_type()
        {
            return components::get_component_type<base_lco_with_value>();
        }
        static void set_component_type(components::component_type type)
        {
            components::set_component_type<base_lco_with_value>(type);
        }

        /// The \a function set_value_nonvirt is called whenever a
        /// \a set_value_action is applied on this LCO instance. This
        /// function just forwards to the virtual function \a set_value, which
        /// is overloaded by the derived concrete LCO.
        ///
        /// \param result [in] The result value to be transferred from the
        ///               remote operation back to this LCO instance.
        void set_value_nonvirt (BOOST_RV_REF(RemoteResult) result)
        {
            set_value(boost::move(result));
        }

        /// The \a function get_result_nonvirt is called whenever a
        /// \a get_result_action is applied on this LCO instance. This
        /// function just forwards to the virtual function \a get_result, which
        /// is overloaded by the derived concrete LCO.
        Result get_value_nonvirt()
        {
            return get_value();
        }

    public:
        /// The \a set_value_action may be used to trigger any LCO instances
        /// while carrying an additional parameter of any type.
        ///
        /// RemoteResult is taken by rvalue ref. This allows for perfect forwarding.
        /// When the action thread function is created, the values are moved into
        /// the calling function. If we took it by const lvalue reference, we
        /// would disable the possibility to further move the result to the
        /// designated destination.
        ///
        /// \param RemoteResult [in] The type of the result to be transferred
        ///               back to this LCO instance.
//         HPX_COMPONENT_DIRECT_ACTION_TPL(base_lco_with_value, set_value_nonvirt,
//             set_value_action);
        typedef hpx::actions::direct_action1<
            base_lco_with_value, lco_set_value, BOOST_RV_REF(RemoteResult),
            &base_lco_with_value::set_value_nonvirt
        > set_value_action;

        /// The \a get_value_action may be used to query the value this LCO
        /// instance exposes as its 'result' value.
//         HPX_COMPONENT_DIRECT_ACTION_TPL(base_lco_with_value, get_value_nonvirt,
//             get_value_action);
        typedef hpx::actions::direct_result_action0<
            base_lco_with_value, Result, lco_get_value,
            &base_lco_with_value::get_value_nonvirt
        > get_value_action;
    };

    /// The base_lco<void> specialization is used whenever the set_event action
    /// for a particular LCO doesn't carry any argument.
    ///
    /// \tparam void This specialization expects no result value and is almost
    ///              completely equivalent to the plain \a base_lco.
    template <>
    class base_lco_with_value<void, void> : public base_lco
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
        typedef base_lco_with_value base_type_holder;
    };
}}

namespace hpx { namespace traits
{
    template <typename Result, typename RemoteResult, typename Enable>
    struct component_type_database<
            hpx::lcos::base_lco_with_value<Result, RemoteResult>
          , Enable
        >
    {
        static components::component_type get()
        {
            return components::component_base_lco_with_value;
        }

        static void set(components::component_type)
        {
            BOOST_ASSERT(false);
        }
    };
}}

///////////////////////////////////////////////////////////////////////////////
// Declaration of serialization support for the base LCO actions
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco::set_event_action, base_set_event_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco::set_exception_action, base_set_exception_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco::connect_action, base_connect_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco::disconnect_action, base_disconnect_action);

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<hpx::naming::gid_type>::set_value_action,
    set_value_action_gid_type);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<hpx::naming::gid_type>::get_value_action,
    get_value_action_gid_type);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<std::vector<hpx::naming::gid_type> >::set_value_action,
    set_value_action_vector_gid_type);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<std::vector<hpx::naming::gid_type> >::get_value_action,
    get_value_action_vector_gid_type);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<hpx::naming::id_type>::set_value_action,
    set_value_action_id_type);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<hpx::naming::id_type>::get_value_action,
    get_value_action_id_type);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<std::vector<hpx::naming::id_type> >::set_value_action,
    set_value_action_vector_id_type);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<std::vector<hpx::naming::id_type> >::get_value_action,
    get_value_action_vector_id_type);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<double>::set_value_action,
    set_value_action_double);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<double>::get_value_action,
    get_value_action_double);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<int>::set_value_action,
    set_value_action_int);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<int>::get_value_action,
    get_value_action_int);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<bool>::set_value_action,
    set_value_action_bool);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<bool>::get_value_action,
    get_value_action_bool);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<hpx::util::section>::set_value_action,
    set_value_action_section);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<hpx::util::section>::get_value_action,
    get_value_action_section);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<hpx::util::unused_type>::set_value_action,
    set_value_action_void);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<hpx::util::unused_type>::get_value_action,
    get_value_action_void);

#endif

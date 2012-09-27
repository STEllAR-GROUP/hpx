//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_BASE_LCO_WITH_VALUE_HPP)
#define HPX_LCOS_BASE_LCO_WITH_VALUE_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/util/ini.hpp>
#include <type_traits>

#include <boost/mpl/bool.hpp>

namespace hpx { namespace lcos
{
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

        // FIXME: gcc complains when the macro is used
//         HPX_COMPONENT_DIRECT_ACTION_TPL(base_lco_with_value, set_value_nonvirt,
//             set_value_action);
        typedef hpx::actions::direct_action1<
            base_lco_with_value, lco_set_value, BOOST_RV_REF(RemoteResult),
            &base_lco_with_value::set_value_nonvirt
        > set_value_action;

        /// The \a get_value_action may be used to query the value this LCO
        /// instance exposes as its 'result' value.

        // FIXME: gcc complains when the macro is used
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
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<hpx::naming::gid_type>::set_value_action,
    set_value_action_gid_type)
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<hpx::naming::gid_type>::get_value_action,
    get_value_action_gid_type)
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<std::vector<hpx::naming::gid_type> >::set_value_action,
    set_value_action_vector_gid_type)
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<std::vector<hpx::naming::gid_type> >::get_value_action,
    get_value_action_vector_gid_type)
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<hpx::naming::id_type>::set_value_action,
    set_value_action_id_type)
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<hpx::naming::id_type>::get_value_action,
    get_value_action_id_type)
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<std::vector<hpx::naming::id_type> >::set_value_action,
    set_value_action_vector_id_type)
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<std::vector<hpx::naming::id_type> >::get_value_action,
    get_value_action_vector_id_type)
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<double>::set_value_action,
    set_value_action_double)
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<double>::get_value_action,
    get_value_action_double)
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<int>::set_value_action,
    set_value_action_int)
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<int>::get_value_action,
    get_value_action_int)
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<bool>::set_value_action,
    set_value_action_bool)
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<bool>::get_value_action,
    get_value_action_bool)
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<hpx::util::section>::set_value_action,
    set_value_action_section)
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<hpx::util::section>::get_value_action,
    get_value_action_section)
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<hpx::util::unused_type>::set_value_action,
    set_value_action_void)
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<hpx::util::unused_type>::get_value_action,
    get_value_action_void)

#endif

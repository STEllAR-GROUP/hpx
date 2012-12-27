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
#include <hpx/runtime/actions/base_lco_continuation.hpp>
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
            base_lco_with_value, BOOST_RV_REF(RemoteResult),
            &base_lco_with_value::set_value_nonvirt
        > set_value_action;

        /// The \a get_value_action may be used to query the value this LCO
        /// instance exposes as its 'result' value.

        // FIXME: gcc complains when the macro is used
//         HPX_COMPONENT_DIRECT_ACTION_TPL(base_lco_with_value, get_value_nonvirt,
//             get_value_action);
        typedef hpx::actions::direct_result_action0<
            base_lco_with_value, Result,
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

#define HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(Value, Name)               \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        hpx::lcos::base_lco_with_value<Value>::set_value_action,                \
        BOOST_PP_CAT(set_value_action_, Name))                                  \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        hpx::lcos::base_lco_with_value<Value>::get_value_action,                \
        BOOST_PP_CAT(get_value_action_, Name))                                  \
    HPX_REGISTER_TYPED_CONTINUATION_DECLARATION(                                \
        Value                                                                   \
      , BOOST_PP_CAT(typed_continuation_, Name))                                \
    HPX_REGISTER_BASE_LCO_CONTINUATION_DECLARATION(                             \
        Value                                                                   \
      , BOOST_PP_CAT(base_lco_continuation_, Name))                             \
/**/

#define HPX_REGISTER_BASE_LCO_WITH_VALUE(Value, Name)                           \
    HPX_REGISTER_ACTION(                                                        \
        hpx::lcos::base_lco_with_value<Value>::set_value_action,                \
        BOOST_PP_CAT(set_value_action_, Name))                                  \
    HPX_REGISTER_ACTION(                                                        \
        hpx::lcos::base_lco_with_value<Value>::get_value_action,                \
        BOOST_PP_CAT(get_value_action_, Name))                                  \
    HPX_REGISTER_TYPED_CONTINUATION(                                            \
        Value                                                                   \
      , BOOST_PP_CAT(typed_continuation_, Name))                                \
    HPX_REGISTER_BASE_LCO_CONTINUATION(                                         \
        Value                                                                   \
      , BOOST_PP_CAT(base_lco_continuation_, Name))                             \
/**/



///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(hpx::naming::gid_type, gid_type)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(std::vector<hpx::naming::gid_type>,
    vector_gid_type)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(hpx::naming::id_type, id_type)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(std::vector<hpx::naming::id_type>,
    vector_id_type)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(float, float)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(double, double)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(boost::int8_t, int8_t)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(boost::uint8_t, uint8_t)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(boost::int16_t, int16_t)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(boost::uint16_t, uint16_t)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(boost::int32_t, int32_t)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(boost::uint32_t, uint32_t)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(boost::int64_t, int64_t)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(boost::uint64_t, uint64_t)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(bool, bool)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(hpx::util::section, section)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(hpx::util::unused_type, void)

#endif

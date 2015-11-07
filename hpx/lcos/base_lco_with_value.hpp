//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_BASE_LCO_WITH_VALUE_HPP)
#define HPX_LCOS_BASE_LCO_WITH_VALUE_HPP

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/util/unused.hpp>
#include <hpx/util/void_guard.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/if.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/type_traits/is_same.hpp>

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
        typedef typename boost::mpl::if_<
            boost::is_same<void, Result>, util::unused_type, Result
        >::type result_type;

    protected:
        /// Destructor, needs to be virtual to allow for clean destruction of
        /// derived objects
        virtual ~base_lco_with_value() {}

        virtual void set_event()
        {
            set_value(RemoteResult());
        }

        virtual void set_value (RemoteResult && result) = 0;

        virtual result_type get_value(error_code& ec = throws) = 0;

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
        void set_value_nonvirt (RemoteResult && result)
        {
            set_value(std::move(result));
        }

        /// The \a function get_result_nonvirt is called whenever a
        /// \a get_result_action is applied on this LCO instance. This
        /// function just forwards to the virtual function \a get_result, which
        /// is overloaded by the derived concrete LCO.
        Result get_value_nonvirt()
        {
            return util::void_guard<Result>(), get_value();
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
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(base_lco_with_value,
            set_value_nonvirt, set_value_action);

        /// The \a get_value_action may be used to query the value this LCO
        /// instance exposes as its 'result' value.
        HPX_DEFINE_COMPONENT_DIRECT_ACTION(base_lco_with_value,
            get_value_nonvirt, get_value_action);
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
    // define component type data base entry generator
    template <typename Result, typename RemoteResult, typename Enable>
    struct component_type_database<
        hpx::lcos::base_lco_with_value<Result, RemoteResult>, Enable>
    {
        static components::component_type get()
        {
            return components::component_base_lco_with_value;
        }

        static void set(components::component_type)
        {
            HPX_ASSERT(false);
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
    HPX_ACTION_USES_MESSAGE_COALESCING_NOTHROW(                                 \
        hpx::lcos::base_lco_with_value<Value>::set_value_action,                \
        "lco_set_value_action", std::size_t(-1), std::size_t(-1))               \
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
/**/

#define HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(                                    \
        Value, Name, ActionIdGet, ActionIdSet)                                  \
    HPX_REGISTER_ACTION_ID(                                                     \
        hpx::lcos::base_lco_with_value<Value>::set_value_action,                \
        BOOST_PP_CAT(set_value_action_, Name), ActionIdSet)                     \
    HPX_REGISTER_ACTION_ID(                                                     \
        hpx::lcos::base_lco_with_value<Value>::get_value_action,                \
        BOOST_PP_CAT(get_value_action_, Name), ActionIdGet)                     \
    HPX_REGISTER_TYPED_CONTINUATION(                                            \
        Value                                                                   \
      , BOOST_PP_CAT(typed_continuation_, Name))                                \
/**/
#define HPX_REGISTER_BASE_LCO_WITH_VALUE_ID2(                                   \
        Value, RemoteValue, Name, ActionIdGet, ActionIdSet)                     \
    typedef hpx::lcos::base_lco_with_value<Value, RemoteValue>                  \
        BOOST_PP_CAT(base_lco_with_value_, Name);                               \
    HPX_REGISTER_ACTION_ID(                                                     \
        BOOST_PP_CAT(base_lco_with_value_, Name)::set_value_action,             \
        BOOST_PP_CAT(set_value_action_, Name), ActionIdSet)                     \
    HPX_REGISTER_ACTION_ID(                                                     \
        BOOST_PP_CAT(base_lco_with_value_, Name)::get_value_action,             \
        BOOST_PP_CAT(get_value_action_, Name), ActionIdGet)                     \
    HPX_REGISTER_TYPED_CONTINUATION(                                            \
        Value                                                                   \
      , BOOST_PP_CAT(typed_continuation_, Name))                                \
/**/


///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(hpx::naming::gid_type, gid_type)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    std::vector<hpx::naming::gid_type>, vector_gid_type)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(hpx::naming::id_type, id_type)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    std::vector<hpx::naming::id_type>, vector_id_type)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    hpx::util::unused_type, hpx_unused_type)
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
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(hpx::util::section, hpx_section)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(std::string, std_string)

#endif

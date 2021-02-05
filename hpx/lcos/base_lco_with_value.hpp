//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2012-2017 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions/transfer_action.hpp>
#include <hpx/actions/transfer_continuation_action.hpp>
#include <hpx/actions_base/component_action.hpp>
#include <hpx/assert.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/components_base/traits/is_component.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/lcos_fwd.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/plugins/parcel/coalescing_message_handler_registration.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/expand.hpp>
#include <hpx/preprocessor/nargs.hpp>
#include <hpx/runtime/components/server/component_base.hpp>
#include <hpx/runtime/components/server/component_heap.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components_fwd.hpp>
#include <hpx/runtime_configuration/ini.hpp>
#include <hpx/type_support/unused.hpp>
#include <hpx/type_support/void_guard.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace lcos
{
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename ComponentTag, typename BaseLco>
        struct base_lco_wrapping_type;

        template <typename BaseLco>
        struct base_lco_wrapping_type<traits::detail::component_tag, BaseLco>
        {
            typedef components::component<BaseLco> type;
        };

        template <typename BaseLco>
        struct base_lco_wrapping_type<traits::detail::managed_component_tag, BaseLco>
        {
            typedef components::managed_component<BaseLco> type;
        };
        /// \endcond
    }

    /// The \a base_lco_with_value class is the common base class for all LCO's
    /// synchronizing on a value.
    /// The \a RemoteResult template argument should be set to the type of the
    /// argument expected for the set_value action.
    ///
    /// \tparam RemoteResult The type of the result value to be carried back
    ///                      to the LCO instance.
    /// \tparam ComponentTag The tag type representing the type of the component
    ///                      (either component_tag or managed_component_tag).
    template <typename Result, typename RemoteResult, typename ComponentTag>
    class base_lco_with_value : public base_lco, public ComponentTag
    {
    protected:
        typedef typename std::conditional<
            std::is_void<Result>::value, util::unused_type, Result
        >::type result_type;

        /// Destructor, needs to be virtual to allow for clean destruction of
        /// derived objects
        virtual ~base_lco_with_value() noexcept {}

        virtual void set_event()
        {
            set_event_nonvirt(std::is_default_constructible<RemoteResult>());
        }

        void set_event_nonvirt(std::false_type)
        {
            // this shouldn't ever be called
            HPX_THROW_EXCEPTION(invalid_status,
                "base_lco_with_value::set_event_nonvirt",
                "attempt to use a non-default-constructible return type with "
                "an action in a context where default-construction would be "
                "required");
        }

        void set_event_nonvirt(std::true_type)
        {
            set_value(RemoteResult());
        }

        virtual void set_value (RemoteResult && result) = 0;

        virtual result_type get_value() = 0;
        virtual result_type get_value(error_code& /*ec*/)
        {
            return get_value();
        }

    public:
        // components must contain a typedef for wrapping_type defining the
        // managed_component type used to encapsulate instances of this
        // component
        typedef typename detail::base_lco_wrapping_type<
                ComponentTag, base_lco_with_value
            >::type wrapping_type;

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

        void set_value_nonvirt (RemoteResult&& result)
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
    template <typename ComponentTag>
    class base_lco_with_value<void, void, ComponentTag>
      : public base_lco, public ComponentTag
    {
    protected:
        /// Destructor, needs to be virtual to allow for clean destruction of
        /// derived objects
        virtual ~base_lco_with_value() {}

    public:
        // components must contain a typedef for wrapping_type defining the
        // managed_component type used to encapsulate instances of this
        // component
        typedef typename detail::base_lco_wrapping_type<ComponentTag,
            base_lco_with_value>::type wrapping_type;
        typedef base_lco_with_value base_type_holder;

#if !defined(HPX_COMPUTE_DEVICE_CODE)
        // refer to base type for the corresponding implementation
        typedef typename base_lco::set_event_action set_value_action;
#endif

        // dummy action definition
        void get_value() {}

        HPX_DEFINE_COMPONENT_DIRECT_ACTION(base_lco_with_value,
            get_value, get_value_action);
    };
}}

namespace hpx { namespace traits
{
    // define component type data base entry generator
    template <typename Result, typename RemoteResult, typename Enable>
    struct component_type_database<
        hpx::lcos::base_lco_with_value<
            Result, RemoteResult, traits::detail::managed_component_tag
        >, Enable>
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

    template <typename Result, typename RemoteResult, typename Enable>
    struct component_type_database<
        hpx::lcos::base_lco_with_value<
            Result, RemoteResult, traits::detail::component_tag
        >, Enable>
    {
        static components::component_type get()
        {
            return components::component_base_lco_with_value_unmanaged;
        }

        static void set(components::component_type)
        {
            HPX_ASSERT(false);
        }
    };
}}
namespace hpx { namespace components {
    namespace detail {
        template <typename Result, typename RemoteResult>
        struct component_heap_impl<
            hpx::components::managed_component<hpx::lcos::base_lco_with_value<
                Result, RemoteResult, traits::detail::managed_component_tag>>>
        {
            typedef void valid;
            typedef hpx::components::managed_component<hpx::lcos::base_lco_with_value<
                Result, RemoteResult, traits::detail::managed_component_tag>>
                component_type;
            HPX_ALWAYS_EXPORT static typename component_type::heap_type& call()
            {
                util::reinitializable_static<typename component_type::heap_type>
                    heap;
                return heap.get();
            }
        };
        template <typename Result, typename RemoteResult>
        struct component_heap_impl<
            hpx::components::managed_component<hpx::lcos::base_lco_with_value<
                Result, RemoteResult, traits::detail::component_tag>>>
        {
            typedef void valid;
            typedef hpx::components::managed_component<hpx::lcos::base_lco_with_value<
                Result, RemoteResult, traits::detail::component_tag>>
                component_type;
            HPX_ALWAYS_EXPORT static
            typename component_type::heap_type& call()
            {
                util::reinitializable_static<typename component_type::heap_type>
                    heap;
                return heap.get();
            }
        };
    }
}}

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(...)                     \
    HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION_(__VA_ARGS__)                \
/**/
#define HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION_(...)                    \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                 \
        HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION_,                        \
        HPX_PP_NARGS(__VA_ARGS__)                                             \
    )(__VA_ARGS__))                                                           \
/**/

// obsolete
#define HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION2(Value, RemoteValue, Name)\
    HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION_3(Value, RemoteValue, Name)  \
/**/

#define HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION_1(Value)                 \
    HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION_4(                           \
        Value, Value, Value, managed_component_tag)                           \
/**/

#define HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION_2(Value, Name)           \
    HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION_4(                           \
        Value, Value, Name, managed_component_tag)                            \
/**/

#define HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION_3(                       \
        Value, RemoteValue, Name)                                             \
    HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION_4(                           \
        Value, RemoteValue, Name, managed_component_tag)                      \
/**/

#define HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION_4(                       \
        Value, RemoteValue, Name, Tag)                                        \
    typedef ::hpx::lcos::base_lco_with_value<Value, RemoteValue,              \
            ::hpx::traits::detail::Tag>                                       \
        HPX_PP_CAT(HPX_PP_CAT(base_lco_with_value_, Name), Tag);              \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        HPX_PP_CAT(HPX_PP_CAT(base_lco_with_value_, Name), Tag)               \
            ::set_value_action,                                               \
        HPX_PP_CAT(HPX_PP_CAT(set_value_action_, Name), Tag))                 \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        HPX_PP_CAT(HPX_PP_CAT(base_lco_with_value_, Name), Tag)               \
            ::get_value_action,                                               \
        HPX_PP_CAT(HPX_PP_CAT(get_value_action_, Name), Tag))                 \
    HPX_ACTION_USES_MESSAGE_COALESCING_NOTHROW_DECLARATION(                   \
        HPX_PP_CAT(HPX_PP_CAT(base_lco_with_value_, Name), Tag)               \
            ::set_value_action,                                               \
        "lco_set_value_action", std::size_t(-1), std::size_t(-1))             \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_BASE_LCO_WITH_VALUE(...)                                 \
    HPX_REGISTER_BASE_LCO_WITH_VALUE_(__VA_ARGS__)                            \
/**/
#define HPX_REGISTER_BASE_LCO_WITH_VALUE_(...)                                \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                 \
        HPX_REGISTER_BASE_LCO_WITH_VALUE_,                                    \
        HPX_PP_NARGS(__VA_ARGS__)                                             \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_BASE_LCO_WITH_VALUE_1(Value)                             \
    HPX_REGISTER_BASE_LCO_WITH_VALUE_4(                                       \
        Value, Value, Value, managed_component_tag)                           \
/**/

#define HPX_REGISTER_BASE_LCO_WITH_VALUE_2(Value, Name)                       \
    HPX_REGISTER_BASE_LCO_WITH_VALUE_4(                                       \
        Value, Value, Name, managed_component_tag)                            \
/**/

#define HPX_REGISTER_BASE_LCO_WITH_VALUE_3(Value, RemoteValue, Name)          \
    HPX_REGISTER_BASE_LCO_WITH_VALUE_4(                                       \
        Value, RemoteValue, Name, managed_component_tag)                      \
/**/

#define HPX_REGISTER_BASE_LCO_WITH_VALUE_4(Value, RemoteValue, Name, Tag)     \
    typedef ::hpx::lcos::base_lco_with_value<Value, RemoteValue,              \
            ::hpx::traits::detail::Tag>                                       \
        HPX_PP_CAT(HPX_PP_CAT(base_lco_with_value_, Name), Tag);              \
    HPX_REGISTER_ACTION(                                                      \
        HPX_PP_CAT(HPX_PP_CAT(base_lco_with_value_, Name), Tag)               \
            ::set_value_action,                                               \
        HPX_PP_CAT(HPX_PP_CAT(set_value_action_, Name), Tag))                 \
    HPX_REGISTER_ACTION(                                                      \
        HPX_PP_CAT(HPX_PP_CAT(base_lco_with_value_, Name), Tag)::             \
            get_value_action,                                                 \
        HPX_PP_CAT(HPX_PP_CAT(get_value_action_, Name), Tag))                 \
    HPX_ACTION_USES_MESSAGE_COALESCING_NOTHROW_DEFINITION(                    \
        HPX_PP_CAT(HPX_PP_CAT(base_lco_with_value_, Name), Tag)               \
            ::set_value_action,                                               \
        "lco_set_value_action", std::size_t(-1), std::size_t(-1))             \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(...)                              \
    HPX_REGISTER_BASE_LCO_WITH_VALUE_ID_(__VA_ARGS__)                         \
/**/
#define HPX_REGISTER_BASE_LCO_WITH_VALUE_ID_(...)                             \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                 \
        HPX_REGISTER_BASE_LCO_WITH_VALUE_ID_,                                 \
        HPX_PP_NARGS(__VA_ARGS__)                                             \
    )(__VA_ARGS__))                                                           \
/**/

// obsolete
#define HPX_REGISTER_BASE_LCO_WITH_VALUE_ID2(                                 \
        Value, RemoteValue, Name, ActionIdGet, ActionIdSet)                   \
    HPX_REGISTER_BASE_LCO_WITH_VALUE_ID_6(                                    \
        Value, RemoteValue, Name, ActionIdGet, ActionIdSet,                   \
        managed_component_tag)                                                \
/**/

#define HPX_REGISTER_BASE_LCO_WITH_VALUE_ID_4(                                \
        Value, Name, ActionIdGet, ActionIdSet)                                \
    HPX_REGISTER_BASE_LCO_WITH_VALUE_ID_6(Value, Value, Name,                 \
        ActionIdGet, ActionIdSet, managed_component_tag)                      \
/**/

#define HPX_REGISTER_BASE_LCO_WITH_VALUE_ID_5(                                \
        Value, RemoteValue, Name, ActionIdGet, ActionIdSet)                   \
    HPX_REGISTER_BASE_LCO_WITH_VALUE_ID_6(Value, RemoteValue, Name,           \
        ActionIdGet, ActionIdSet, managed_component_tag)                      \
/**/

#define HPX_REGISTER_BASE_LCO_WITH_VALUE_ID_6(                                \
    Value, RemoteValue, Name, ActionIdGet, ActionIdSet, Tag)                  \
    typedef ::hpx::lcos::base_lco_with_value<Value, RemoteValue,              \
            ::hpx::traits::detail::Tag>                                       \
        HPX_PP_CAT(HPX_PP_CAT(base_lco_with_value_, Name), Tag);              \
    HPX_REGISTER_ACTION_ID(                                                   \
        HPX_PP_CAT(HPX_PP_CAT(base_lco_with_value_, Name), Tag)               \
            ::set_value_action,                                               \
        HPX_PP_CAT(HPX_PP_CAT(set_value_action_, Name), Tag), ActionIdSet)    \
    HPX_REGISTER_ACTION_ID(                                                   \
        HPX_PP_CAT(HPX_PP_CAT(base_lco_with_value_, Name), Tag)               \
            ::get_value_action,                                               \
        HPX_PP_CAT(HPX_PP_CAT(get_value_action_, Name), Tag), ActionIdGet)    \
    HPX_ACTION_USES_MESSAGE_COALESCING_NOTHROW_DEFINITION(                    \
        HPX_PP_CAT(HPX_PP_CAT(base_lco_with_value_, Name), Tag)               \
            ::set_value_action,                                               \
        "lco_set_value_action", std::size_t(-1), std::size_t(-1))             \
/**/

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(hpx::naming::gid_type, gid_type)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    std::vector<hpx::naming::gid_type>, vector_gid_type)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    hpx::naming::id_type, hpx::naming::gid_type, id_type)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    hpx::naming::id_type, naming_id_type)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    std::vector<hpx::naming::id_type>, std::vector<hpx::naming::gid_type>,
    vector_id_gid_type)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    std::vector<hpx::naming::id_type>, vector_id_type)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    hpx::util::unused_type, hpx_unused_type)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(float)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(double)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(std::int8_t, int8_t)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(std::uint8_t, uint8_t)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(std::int16_t, int16_t)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(std::uint16_t, uint16_t)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(std::int32_t, int32_t)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(std::uint32_t, uint32_t)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(std::int64_t, int64_t)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(std::uint64_t, uint64_t)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(bool)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(std::vector<bool>, vector_bool_type)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    std::vector<std::uint32_t>, vector_std_uint32_type)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(hpx::util::section, hpx_section)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(std::string, std_string)

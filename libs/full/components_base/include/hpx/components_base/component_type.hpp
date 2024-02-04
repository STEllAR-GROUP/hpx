//  Copyright (c) 2007-2024 Hartmut Kaiser
//  Copyright (c)      2017 Thomas Heller
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/components_base/components_base_fwd.hpp>
#include <hpx/components_base/traits/component_type_database.hpp>
#include <hpx/functional/move_only_function.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/naming_base/gid_type.hpp>
#include <hpx/naming_base/naming_base.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/expand.hpp>
#include <hpx/preprocessor/nargs.hpp>
#include <hpx/preprocessor/stringize.hpp>
#include <hpx/preprocessor/strip_parens.hpp>
#include <hpx/thread_support/atomic_count.hpp>

#include <cstdint>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::components {

    // declared in hpx/modules/naming.hpp
    // using component_type = std::int32_t;

    enum class component_enum_type : naming::component_type
    {
        invalid = naming::address::component_invalid,

        // Runtime support component (provides system services such as
        // component creation, etc.). One per locality.
        runtime_support = 0,

        // Pseudo-component to be used for plain actions
        plain_function = 1,

        // Base component for LCOs that do not produce a value.
        base_lco = 2,

        // Base component for LCOs that produce values.
        base_lco_with_value_unmanaged = 3,

        // (Managed) base component for LCOs that produce values.
        base_lco_with_value = 4,

        // Synchronization latch, barrier, and flex_barrier LCOs.
        latch = ((5 << 10) | base_lco_with_value),
        barrier = ((6 << 10) | base_lco),

        // An LCO representing a value which may not have been computed yet.
        promise = ((7 << 10) | base_lco_with_value),

        // AGAS locality services.
        agas_locality_namespace = 8,

        // AGAS primary address resolution services.
        agas_primary_namespace = 9,

        // AGAS global type system.
        agas_component_namespace = 10,

        // AGAS symbolic naming services.
        agas_symbol_namespace = 11,

        last,
        first_dynamic = last,
    };

    constexpr naming::component_type to_int(component_enum_type t) noexcept
    {
        return static_cast<naming::component_type>(t);
    }

#define HPX_COMPONENT_ENUM_TYPE_ENUM_DEPRECATION_MSG                           \
    "The unscoped hpx::components::component_enum_type names are deprecated. " \
    "Please use hpx::components::component_enum_type::<value> instead."

    HPX_DEPRECATED_V(1, 10, HPX_COMPONENT_ENUM_TYPE_ENUM_DEPRECATION_MSG)
    inline constexpr component_enum_type component_invalid =
        component_enum_type::invalid;
    HPX_DEPRECATED_V(1, 10, HPX_COMPONENT_ENUM_TYPE_ENUM_DEPRECATION_MSG)
    inline constexpr component_enum_type component_runtime_support =
        component_enum_type::runtime_support;
    HPX_DEPRECATED_V(1, 10, HPX_COMPONENT_ENUM_TYPE_ENUM_DEPRECATION_MSG)
    inline constexpr component_enum_type component_plain_function =
        component_enum_type::plain_function;
    HPX_DEPRECATED_V(1, 10, HPX_COMPONENT_ENUM_TYPE_ENUM_DEPRECATION_MSG)
    inline constexpr component_enum_type component_base_lco =
        component_enum_type::base_lco;
    HPX_DEPRECATED_V(1, 10, HPX_COMPONENT_ENUM_TYPE_ENUM_DEPRECATION_MSG)
    inline constexpr component_enum_type
        component_base_lco_with_value_unmanaged =
            component_enum_type::base_lco_with_value_unmanaged;
    HPX_DEPRECATED_V(1, 10, HPX_COMPONENT_ENUM_TYPE_ENUM_DEPRECATION_MSG)
    inline constexpr component_enum_type component_base_lco_with_value =
        component_enum_type::base_lco_with_value;
    HPX_DEPRECATED_V(1, 10, HPX_COMPONENT_ENUM_TYPE_ENUM_DEPRECATION_MSG)
    inline constexpr component_enum_type component_latch =
        component_enum_type::latch;
    HPX_DEPRECATED_V(1, 10, HPX_COMPONENT_ENUM_TYPE_ENUM_DEPRECATION_MSG)
    inline constexpr component_enum_type component_barrier =
        component_enum_type::barrier;
    HPX_DEPRECATED_V(1, 10, HPX_COMPONENT_ENUM_TYPE_ENUM_DEPRECATION_MSG)
    inline constexpr component_enum_type component_promise =
        component_enum_type::promise;
    HPX_DEPRECATED_V(1, 10, HPX_COMPONENT_ENUM_TYPE_ENUM_DEPRECATION_MSG)
    inline constexpr component_enum_type component_agas_locality_namespace =
        component_enum_type::agas_locality_namespace;
    HPX_DEPRECATED_V(1, 10, HPX_COMPONENT_ENUM_TYPE_ENUM_DEPRECATION_MSG)
    inline constexpr component_enum_type component_agas_primary_namespace =
        component_enum_type::agas_primary_namespace;
    HPX_DEPRECATED_V(1, 10, HPX_COMPONENT_ENUM_TYPE_ENUM_DEPRECATION_MSG)
    inline constexpr component_enum_type component_agas_component_namespace =
        component_enum_type::agas_component_namespace;
    HPX_DEPRECATED_V(1, 10, HPX_COMPONENT_ENUM_TYPE_ENUM_DEPRECATION_MSG)
    inline constexpr component_enum_type component_agas_symbol_namespace =
        component_enum_type::agas_symbol_namespace;
    HPX_DEPRECATED_V(1, 10, HPX_COMPONENT_ENUM_TYPE_ENUM_DEPRECATION_MSG)
    inline constexpr component_enum_type component_last =
        component_enum_type::last;
    HPX_DEPRECATED_V(1, 10, HPX_COMPONENT_ENUM_TYPE_ENUM_DEPRECATION_MSG)
    inline constexpr component_enum_type component_first_dynamic =
        component_enum_type::first_dynamic;

#undef HPX_COMPONENT_ENUM_TYPE_ENUM_DEPRECATION_MSG

    enum class factory_state : std::uint8_t
    {
        enabled = 0,
        disabled = 1,
        check = 2
    };

    constexpr int to_int(factory_state t) noexcept
    {
        return static_cast<int>(t);
    }

#define HPX_FACTORY_STATE_ENUM_DEPRECATION_MSG                                 \
    "The unscoped hpx::components::factory_state_enum names are deprecated. "  \
    "Please use hpx::components::factory_state::<value> instead."

    HPX_DEPRECATED_V(1, 10, HPX_FACTORY_STATE_ENUM_DEPRECATION_MSG)
    inline constexpr factory_state factory_enabled = factory_state::enabled;
    HPX_DEPRECATED_V(1, 10, HPX_FACTORY_STATE_ENUM_DEPRECATION_MSG)
    inline constexpr factory_state factory_disabled = factory_state::disabled;
    HPX_DEPRECATED_V(1, 10, HPX_FACTORY_STATE_ENUM_DEPRECATION_MSG)
    inline constexpr factory_state factory_check = factory_state::check;

#undef HPX_FACTORY_STATE_ENUM_DEPRECATION_MSG

    // access data related to component instance counts
    HPX_EXPORT bool& enabled(component_type type);
    HPX_EXPORT util::atomic_count& instance_count(component_type type);

    using component_deleter_type = void (*)(
        hpx::naming::gid_type const&, hpx::naming::address const&);
    HPX_EXPORT component_deleter_type& deleter(component_type type);

    HPX_EXPORT bool enumerate_instance_counts(
        hpx::move_only_function<bool(component_type)> const& f);

    /// \brief Return the string representation for a given component type id
    HPX_EXPORT std::string get_component_type_name(component_type type);

    /// The lower short word of the component type is the type of the component
    /// exposing the actions.
    constexpr component_type get_base_type(component_type t) noexcept
    {
        return static_cast<component_type>(t & 0x3FF);
    }

    /// The upper short word of the component is the actual component type
    constexpr component_type get_derived_type(component_type t) noexcept
    {
        return static_cast<component_type>((t >> 10) & 0x3FF);
    }

    /// A component derived from a base component exposing the actions needs to
    /// have a specially formatted component type.
    constexpr component_type derived_component_type(
        component_type derived, component_type base) noexcept
    {
        return static_cast<component_type>(derived << 10 | base);
    }

    /// \brief Verify the two given component types are matching (compatible)
    constexpr bool types_are_compatible(
        component_type lhs, component_type rhs) noexcept
    {
        // don't compare types if one of them is unknown
        if (to_int(component_enum_type::invalid) == rhs ||
            to_int(component_enum_type::invalid) == lhs)
        {
            return true;    // no way of telling, so we assume the best :-P
        }

        // don't compare types if one of them is component_runtime_support
        if (to_int(component_enum_type::runtime_support) == rhs ||
            to_int(component_enum_type::runtime_support) == lhs)
        {
            return true;
        }

        component_type const lhs_base = get_base_type(lhs);
        component_type const rhs_base = get_base_type(rhs);

        if (lhs_base == rhs_base)
        {
            return true;
        }

        // special case for lco's
        if (lhs_base == to_int(component_enum_type::base_lco) &&
            (rhs_base ==
                    to_int(
                        component_enum_type::base_lco_with_value_unmanaged) ||
                rhs_base == to_int(component_enum_type::base_lco_with_value)))
        {
            return true;
        }

        if (rhs_base == to_int(component_enum_type::base_lco) &&
            (lhs_base ==
                    to_int(
                        component_enum_type::base_lco_with_value_unmanaged) ||
                lhs_base == to_int(component_enum_type::base_lco_with_value)))
        {
            return true;
        }

        if (lhs_base ==
                to_int(component_enum_type::base_lco_with_value_unmanaged) &&
            rhs_base == to_int(component_enum_type::base_lco_with_value))
        {
            return true;
        }

        if (lhs_base == to_int(component_enum_type::base_lco_with_value) &&
            rhs_base ==
                to_int(component_enum_type::base_lco_with_value_unmanaged))
        {
            return true;
        }

        return false;
    }

    namespace detail {

        // Resolve the type from AGAS
        HPX_EXPORT component_type get_agas_component_type(
            char const* name, char const* base_name, component_type, bool);
    }    // namespace detail

    // Returns the (unique) name for a given component
    template <typename Component, typename Enable = void>
    HPX_ALWAYS_EXPORT char const* get_component_name() noexcept;

    // Returns the (unique) name of the base component. If there is none,
    // nullptr is returned
    template <typename Component, typename Enable = void>
    HPX_ALWAYS_EXPORT char const* get_component_base_name() noexcept;

    template <typename Component>
    component_type get_component_type() noexcept
    {
        return traits::component_type_database<Component>::get();
    }

    template <typename Component>
    void set_component_type(component_type type)
    {
        traits::component_type_database<Component>::set(type);
    }
}    // namespace hpx::components

///////////////////////////////////////////////////////////////////////////////
#define HPX_DEFINE_GET_COMPONENT_TYPE(component)                               \
    namespace hpx::traits {                                                    \
        template <>                                                            \
        HPX_ALWAYS_EXPORT components::component_type                           \
        component_type_database<component>::get() noexcept                     \
        {                                                                      \
            return value;                                                      \
        }                                                                      \
        template <>                                                            \
        HPX_ALWAYS_EXPORT void component_type_database<component>::set(        \
            components::component_type t)                                      \
        {                                                                      \
            value = t;                                                         \
        }                                                                      \
    }                                                                          \
    /**/

#define HPX_DEFINE_GET_COMPONENT_TYPE_TEMPLATE(template_, component)           \
    namespace hpx::traits {                                                    \
        HPX_PP_STRIP_PARENS(template_)                                         \
        struct component_type_database<HPX_PP_STRIP_PARENS(component)>         \
        {                                                                      \
            static components::component_type value;                           \
                                                                               \
            HPX_ALWAYS_EXPORT static components::component_type get() noexcept \
            {                                                                  \
                return value;                                                  \
            }                                                                  \
            HPX_ALWAYS_EXPORT static void set(components::component_type t)    \
            {                                                                  \
                value = t;                                                     \
            }                                                                  \
        };                                                                     \
                                                                               \
        HPX_PP_STRIP_PARENS(template_)                                         \
        components::component_type                                             \
            component_type_database<HPX_PP_STRIP_PARENS(component)>::value =   \
                to_int(hpx::components::component_enum_type::invalid);         \
    }                                                                          \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(component, type)                  \
    namespace hpx::traits {                                                    \
        template <>                                                            \
        HPX_ALWAYS_EXPORT components::component_type                           \
        component_type_database<component>::get() noexcept                     \
        {                                                                      \
            return type;                                                       \
        }                                                                      \
        template <>                                                            \
        HPX_ALWAYS_EXPORT void component_type_database<component>::set(        \
            components::component_type)                                        \
        {                                                                      \
            HPX_ASSERT(false);                                                 \
        }                                                                      \
    }                                                                          \
    /**/

#define HPX_DEFINE_COMPONENT_NAME(...) HPX_DEFINE_COMPONENT_NAME_(__VA_ARGS__)

#define HPX_DEFINE_COMPONENT_NAME_(...)                                        \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                  \
        HPX_DEFINE_COMPONENT_NAME_, HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))   \
    /**/

#define HPX_DEFINE_COMPONENT_NAME_2(Component, name)                           \
    namespace hpx::components {                                                \
        template <>                                                            \
        HPX_ALWAYS_EXPORT char const*                                          \
        get_component_name<Component, void>() noexcept                         \
        {                                                                      \
            return HPX_PP_STRINGIZE(name);                                     \
        }                                                                      \
        template <>                                                            \
        HPX_ALWAYS_EXPORT char const*                                          \
        get_component_base_name<Component, void>() noexcept                    \
        {                                                                      \
            return nullptr;                                                    \
        }                                                                      \
    }                                                                          \
    /**/

#define HPX_DEFINE_COMPONENT_NAME_3(Component, name, base_name)                \
    namespace hpx::components {                                                \
        template <>                                                            \
        HPX_ALWAYS_EXPORT char const*                                          \
        get_component_name<Component, void>() noexcept                         \
        {                                                                      \
            return HPX_PP_STRINGIZE(name);                                     \
        }                                                                      \
        template <>                                                            \
        HPX_ALWAYS_EXPORT char const*                                          \
        get_component_base_name<Component, void>() noexcept                    \
        {                                                                      \
            return base_name;                                                  \
        }                                                                      \
    }                                                                          \
    /**/

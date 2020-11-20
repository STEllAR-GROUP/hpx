//  Copyright (c) 2007-2019 Hartmut Kaiser
//  Copyright (c)      2017 Thomas Heller
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/components_base/traits/component_type_database.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/naming_base/gid_type.hpp>
#include <hpx/naming_base/naming_base.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/expand.hpp>
#include <hpx/preprocessor/nargs.hpp>
#include <hpx/preprocessor/stringize.hpp>
#include <hpx/preprocessor/strip_parens.hpp>
#include <hpx/thread_support/atomic_count.hpp>
#include <hpx/type_support/decay.hpp>

#include <cstdint>
#include <iosfwd>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components {

    // declared in hpx/modules/naming.hpp
    // using component_type = std::int32_t;

    enum component_enum_type
    {
        component_invalid = naming::address::component_invalid,

        // Runtime support component (provides system services such as
        // component creation, etc.). One per locality.
        component_runtime_support = 0,

        // Pseudo-component to be used for plain actions
        component_plain_function = 1,

        // Base component for LCOs that do not produce a value.
        component_base_lco = 2,

        // Base component for LCOs that produce values.
        component_base_lco_with_value_unmanaged = 3,

        // (Managed) base component for LCOs that produce values.
        component_base_lco_with_value = 4,

        // Synchronization latch, barrier, and flex_barrier LCOs.
        component_latch = ((5 << 10) | component_base_lco_with_value),
        component_barrier = ((6 << 10) | component_base_lco),

        // An LCO representing a value which may not have been computed yet.
        component_promise = ((7 << 10) | component_base_lco_with_value),

        // AGAS locality services.
        component_agas_locality_namespace = 8,

        // AGAS primary address resolution services.
        component_agas_primary_namespace = 9,

        // AGAS global type system.
        component_agas_component_namespace = 10,

        // AGAS symbolic naming services.
        component_agas_symbol_namespace = 11,

        component_last,
        component_first_dynamic = component_last,

        // Force this enum type to be at least 20 bits.
        component_upper_bound = 0xfffffL    //-V112
    };

    enum factory_state_enum
    {
        factory_enabled = 0,
        factory_disabled = 1,
        factory_check = 2
    };

    // access data related to component instance counts
    HPX_EXPORT bool& enabled(component_type type);
    HPX_EXPORT util::atomic_count& instance_count(component_type type);
    typedef void (*component_deleter_type)(
        hpx::naming::gid_type const&, hpx::naming::address const&);
    HPX_EXPORT component_deleter_type& deleter(component_type type);

    HPX_EXPORT bool enumerate_instance_counts(
        util::unique_function_nonser<bool(component_type)> const& f);

    /// \brief Return the string representation for a given component type id
    HPX_EXPORT std::string const get_component_type_name(component_type type);

    /// The lower short word of the component type is the type of the component
    /// exposing the actions.
    inline component_type get_base_type(component_type t)
    {
        return component_type(t & 0x3FF);
    }

    /// The upper short word of the component is the actual component type
    inline component_type get_derived_type(component_type t)
    {
        return component_type((t >> 10) & 0x3FF);
    }

    /// A component derived from a base component exposing the actions needs to
    /// have a specially formatted component type.
    inline component_type derived_component_type(
        component_type derived, component_type base)
    {
        return component_type(derived << 10 | base);
    }

    /// \brief Verify the two given component types are matching (compatible)
    inline bool types_are_compatible(component_type lhs, component_type rhs)
    {
        // don't compare types if one of them is unknown
        if (component_invalid == rhs || component_invalid == lhs)
            return true;    // no way of telling, so we assume the best :-P

        // don't compare types if one of them is component_runtime_support
        if (component_runtime_support == rhs ||
            component_runtime_support == lhs)
            return true;

        component_type lhs_base = get_base_type(lhs);
        component_type rhs_base = get_base_type(rhs);

        if (lhs_base == rhs_base)
            return true;

        // special case for lco's
        if (lhs_base == component_base_lco &&
            (rhs_base == component_base_lco_with_value_unmanaged ||
                rhs_base == component_base_lco_with_value))
        {
            return true;
        }

        if (rhs_base == component_base_lco &&
            (lhs_base == component_base_lco_with_value_unmanaged ||
                lhs_base == component_base_lco_with_value))
        {
            return true;
        }

        if (lhs_base == component_base_lco_with_value_unmanaged &&
            rhs_base == component_base_lco_with_value)
        {
            return true;
        }

        if (lhs_base == component_base_lco_with_value &&
            rhs_base == component_base_lco_with_value_unmanaged)
        {
            return true;
        }

        return false;
    }

    namespace detail {
        // Resolve the type from AGAS
        HPX_EXPORT component_type get_agas_component_type(
            const char* name, const char* base_name, component_type, bool);
    }    // namespace detail

    // Returns the (unique) name for a given component
    template <typename Component, typename Enable = void>
    constexpr char const* get_component_name();

    // Returns the (unique) name of the base component. If there is none,
    // nullptr is returned
    template <typename Component, typename Enable = void>
    constexpr const char* get_component_base_name();

    template <typename Component>
    inline void set_component_type(component_type type)
    {
        traits::component_type_database<Component>::set(type);
    }

    template <typename Component>
    inline component_type get_component_type()
    {
        return traits::component_type_database<Component>::get();
    }

}}    // namespace hpx::components

namespace hpx { namespace naming {

    // this is defined in this module as its implementation relies on
    // components::get_component_type_name()
    HPX_EXPORT std::ostream& operator<<(std::ostream&, address const&);
}}    // namespace hpx::naming

///////////////////////////////////////////////////////////////////////////////
#define HPX_DEFINE_GET_COMPONENT_TYPE(component)                               \
    namespace hpx { namespace traits {                                         \
            template <>                                                        \
            HPX_ALWAYS_EXPORT components::component_type                       \
            component_type_database<component>::get()                          \
            {                                                                  \
                return value;                                                  \
            }                                                                  \
            template <>                                                        \
            HPX_ALWAYS_EXPORT void component_type_database<component>::set(    \
                components::component_type t)                                  \
            {                                                                  \
                value = t;                                                     \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    /**/

#define HPX_DEFINE_GET_COMPONENT_TYPE_TEMPLATE(template_, component)           \
    namespace hpx { namespace traits {                                         \
            HPX_PP_STRIP_PARENS(template_)                                     \
            struct component_type_database<HPX_PP_STRIP_PARENS(component)>     \
            {                                                                  \
                static components::component_type value;                       \
                                                                               \
                HPX_ALWAYS_EXPORT static components::component_type get()      \
                {                                                              \
                    return value;                                              \
                }                                                              \
                HPX_ALWAYS_EXPORT static void set(                             \
                    components::component_type t)                              \
                {                                                              \
                    value = t;                                                 \
                }                                                              \
            };                                                                 \
                                                                               \
            HPX_PP_STRIP_PARENS(template_)                                     \
            components::component_type component_type_database<                \
                HPX_PP_STRIP_PARENS(component)>::value =                       \
                components::component_invalid;                                 \
        }                                                                      \
    }                                                                          \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(component, type)                  \
    namespace hpx { namespace traits {                                         \
            typedef void static_;                                              \
            template <>                                                        \
            HPX_ALWAYS_EXPORT components::component_type                       \
            component_type_database<component>::get()                          \
            {                                                                  \
                return type;                                                   \
            }                                                                  \
            template <>                                                        \
            HPX_ALWAYS_EXPORT void component_type_database<component>::set(    \
                components::component_type)                                    \
            {                                                                  \
                HPX_ASSERT(false);                                             \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    /**/

#define HPX_DEFINE_COMPONENT_NAME(...) HPX_DEFINE_COMPONENT_NAME_(__VA_ARGS__)

#define HPX_DEFINE_COMPONENT_NAME_(...)                                        \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                  \
        HPX_DEFINE_COMPONENT_NAME_, HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))   \
    /**/

#define HPX_DEFINE_COMPONENT_NAME_2(Component, name)                           \
    namespace hpx { namespace components {                                     \
            template <>                                                        \
            constexpr char const* get_component_name<Component, void>()        \
            {                                                                  \
                return HPX_PP_STRINGIZE(name);                                 \
            }                                                                  \
            template <>                                                        \
            constexpr char const* get_component_base_name<Component, void>()   \
            {                                                                  \
                return nullptr;                                                \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    /**/

#define HPX_DEFINE_COMPONENT_NAME_3(Component, name, base_name)                \
    namespace hpx { namespace components {                                     \
            template <>                                                        \
            constexpr char const* get_component_name<Component, void>()        \
            {                                                                  \
                return HPX_PP_STRINGIZE(name);                                 \
            }                                                                  \
            template <>                                                        \
            constexpr char const* get_component_base_name<Component, void>()   \
            {                                                                  \
                return base_name;                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    /**/

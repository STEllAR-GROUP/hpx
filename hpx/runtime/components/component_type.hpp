//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_COMPONENTS_COMPONENT_TYPE_HPP
#define HPX_RUNTIME_COMPONENTS_COMPONENT_TYPE_HPP

#include <hpx/config.hpp>
#include <hpx/traits/component_type_database.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/detail/pp/strip_parens.hpp>

#include <cstdint>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    typedef std::int32_t component_type;

    enum component_enum_type
    {
        component_invalid = -1,

        // Runtime support component (provides system services such as
        // component creation, etc.). One per locality.
        component_runtime_support = 0,

        // Pseudo-component to be used for plain actions
        component_plain_function = 1,

        // Pseudo-component for direct access to local virtual memory.
        component_memory = 2,

        // Generic memory blocks.
        component_memory_block = 3,

        // Base component for LCOs that do not produce a value.
        component_base_lco = 4,

        // Base component for LCOs that produce values.
        component_base_lco_with_value_unmanaged = 5,

        // (Managed) base component for LCOs that produce values.
        component_base_lco_with_value = 6,

        // Synchronization latch, barrier, and flex_barrier LCOs.
        component_latch = ((7 << 16) | component_base_lco_with_value),
        component_barrier = ((8 << 16) | component_base_lco),
        component_flex_barrier = ((9 << 16) | component_base_lco),

        // An LCO representing a value which may not have been computed yet.
        component_promise = ((10 << 16) | component_base_lco_with_value),

        // AGAS locality services.
        component_agas_locality_namespace = 11,

        // AGAS primary address resolution services.
        component_agas_primary_namespace = 12,

        // AGAS global type system.
        component_agas_component_namespace = 13,

        // AGAS symbolic naming services.
        component_agas_symbol_namespace = 14,

        component_last,
        component_first_dynamic = component_last,

        // Force this enum type to be at least 32 bits.
        component_upper_bound = 0x7fffffffL //-V112
    };

    enum factory_state_enum
    {
        factory_enabled  = 0,
        factory_disabled = 1,
        factory_check    = 2
    };

    /// \brief Return the string representation for a given component type id
    HPX_EXPORT std::string const get_component_type_name(component_type type);

    /// The lower short word of the component type is the type of the component
    /// exposing the actions.
    inline component_type get_base_type(component_type t)
    {
        return component_type(t & 0xFFFF);
    }

    /// The upper short word of the component is the actual component type
    inline component_type get_derived_type(component_type t)
    {
        return component_type((t >> 16) & 0xFFFF);
    }

    /// A component derived from a base component exposing the actions needs to
    /// have a specially formatted component type.
    inline component_type
    derived_component_type(component_type derived, component_type base)
    {
        return component_type(derived << 16 | base);
    }

    /// \brief Verify the two given component types are matching (compatible)
    inline bool types_are_compatible(component_type lhs, component_type rhs)
    {
        // don't compare types if one of them is unknown
        if (component_invalid == rhs || component_invalid == lhs)
            return true;    // no way of telling, so we assume the best :-P

        // don't compare types if one of them is component_runtime_support
        if (component_runtime_support == rhs || component_runtime_support == lhs)
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

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    inline component_type get_component_type()
    {
        return traits::component_type_database<Component>::get();
    }

    template <typename Component>
    inline void set_component_type(component_type type)
    {
        traits::component_type_database<Component>::set(type);
    }
}}

///////////////////////////////////////////////////////////////////////////////
#define HPX_DEFINE_GET_COMPONENT_TYPE(component)                              \
    namespace hpx { namespace traits                                          \
    {                                                                         \
        template <> HPX_ALWAYS_EXPORT                                         \
        components::component_type component_type_database< component>::get() \
            { return value; }                                                 \
        template <> HPX_ALWAYS_EXPORT                                         \
        void component_type_database< component>::set(                        \
            components::component_type t) { value = t; }                      \
    }}                                                                        \
/**/

#define HPX_DEFINE_GET_COMPONENT_TYPE_TEMPLATE(template_, component)          \
    namespace hpx { namespace traits                                          \
    {                                                                         \
        HPX_PP_STRIP_PARENS(template_)                                        \
        struct component_type_database<HPX_PP_STRIP_PARENS(component) >       \
        {                                                                     \
            static components::component_type value;                          \
                                                                              \
            HPX_ALWAYS_EXPORT static components::component_type get()         \
                { return value; }                                             \
            HPX_ALWAYS_EXPORT static void set(components::component_type t)   \
                { value = t; }                                                \
        };                                                                    \
                                                                              \
        HPX_PP_STRIP_PARENS(template_) components::component_type             \
        component_type_database<HPX_PP_STRIP_PARENS(component) >::value =     \
            components::component_invalid;                                    \
    }}                                                                        \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(component, type)                 \
    namespace hpx { namespace traits                                          \
    {                                                                         \
        template <> HPX_ALWAYS_EXPORT                                         \
        components::component_type component_type_database< component>::get() \
            { return type; }                                                  \
        template <> HPX_ALWAYS_EXPORT                                         \
        void component_type_database< component>::set(components::component_type)\
            { HPX_ASSERT(false); }                                            \
    }}                                                                        \
/**/

#endif /*HPX_RUNTIME_COMPONENTS_COMPONENT_TYPE_HPP*/

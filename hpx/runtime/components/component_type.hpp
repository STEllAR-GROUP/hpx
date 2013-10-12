//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENT_COMPONENT_TYPE_MAR_26_2008_1058AM)
#define HPX_COMPONENT_COMPONENT_TYPE_MAR_26_2008_1058AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/traits/component_type_database.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>
#if defined(HPX_HAVE_SECURITY)
#include <hpx/components/security/capability.hpp>
#endif

#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/cstdint.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
        enum component_enum_type
        {
            component_invalid = -1,

            // Runtime support component (provides system services such as
            // component creation, etc). One per locality.
            component_runtime_support = 0,

            // Pseudo-component for direct access to local virtual memory.
            component_memory = 1,

            // Generic memory blocks.
            component_memory_block = 2,

            // Base component for LCOs that do not produce a value.
            component_base_lco = 3,

            // Base component for LCOs that do produce values.
            component_base_lco_with_value = 4,

            // Synchronization barrier LCO.
            component_barrier = ((5 << 16) | component_base_lco),

            // An LCO representing a value which may not have been computed yet.
            component_promise = ((6 << 16) | component_base_lco_with_value),
            gid_promise = ((7 << 16) | component_base_lco_with_value),
            vector_gid_romise = ((8 << 16) | component_base_lco_with_value),
            id_promise = ((9 << 16) | component_base_lco_with_value),
            id_gid_promise = ((10 << 16) | component_base_lco_with_value),
            vector_id_promise = ((11 << 16) | component_base_lco_with_value),
            id_vector_gid_vector_promise = ((12 << 16) | component_base_lco_with_value),
            void_promise = ((13 << 16) | component_base_lco_with_value),
            float_promise = ((14 << 16) | component_base_lco_with_value),
            double_promise = ((15 << 16) | component_base_lco_with_value),
            int8_t_promise = ((16 << 16) | component_base_lco_with_value),
            uint8_t_promise = ((17 << 16) | component_base_lco_with_value),
            int16_t_promise = ((18 << 16) | component_base_lco_with_value),
            uint16_t_promise = ((19 << 16) | component_base_lco_with_value),
            int32_t_promise = ((20 << 16) | component_base_lco_with_value),
            uint32_t_promise = ((21 << 16) | component_base_lco_with_value),
            int64_t_promise = ((22 << 16) | component_base_lco_with_value),
            uint64_t_promise = ((23 << 16) | component_base_lco_with_value),
            string_promise = ((24 << 16) | component_base_lco_with_value),
            bool_promise = ((25 << 16) | component_base_lco_with_value),
            section_promise = ((26 << 16) | component_base_lco_with_value),
            counter_info_promise = ((27 << 16) | component_base_lco_with_value),
            counter_value_promise = ((28 << 16) | component_base_lco_with_value),
            agas_response_promise = ((29 << 16) | component_base_lco_with_value),
            agas_response_vector_promise = ((30 << 16) | component_base_lco_with_value),
            id_type_response_promise = ((31 << 16) | component_base_lco_with_value),
            bool_response_promise = ((32 << 16) | component_base_lco_with_value),
            uint32_t_response_promise = ((33 << 16) | component_base_lco_with_value),
            uint32_t_vector_response_promise = ((34 << 16) | component_base_lco_with_value),
            locality_vector_response_promise = ((35 << 16) | component_base_lco_with_value),
            memory_data_promise = ((36 << 16) | component_base_lco_with_value),
            factory_locality_promise = ((37 << 16) | component_base_lco_with_value),

            // AGAS locality services.
            component_agas_locality_namespace = 38,

            // AGAS primary address resolution services.
            component_agas_primary_namespace = 39,

            // AGAS global type system.
            component_agas_component_namespace = 40,

            // AGAS symbolic naming services.
            component_agas_symbol_namespace = 41,

#if defined(HPX_HAVE_SODIUM)
            // root CA, subordinate CA
            signed_certificate_promise = ((42 << 16) | component_base_lco_with_value),
            component_root_certificate_authority = 43,
            component_subordinate_certificate_authority = 44,
#endif

            component_last,
            component_first_dynamic = component_last,

            // Force this enum type to be at least 32 bits.
            component_upper_bound = 0x7fffffffL //-V112
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

        // special case for lco's
        if ((lhs_base == component_base_lco && rhs_base == component_base_lco_with_value) ||
            (rhs_base == component_base_lco && lhs_base == component_base_lco_with_value))
        {
            return true;
        }
        return lhs_base == rhs_base;
    }

#if defined(HPX_HAVE_SECURITY)
    inline components::security::capability default_component_creation_capabilities(
        components::security::traits::capability<>::capabilities caps)
    {
        using namespace components::security;

        // if we're asked for required capabilities related to creating
        // an instance of this component then require 'write' capabilities
        if (caps & traits::capability<>::capability_create_component)
        {
            return capability(traits::capability<>::capability_non_const);
        }

        // otherwise require no capabilities
        return capability();
    }
#endif

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

    ///////////////////////////////////////////////////////////////////////////
    enum factory_property
    {
        factory_invalid = -1,
        factory_none = 0,                   ///< The factory has no special properties
        factory_is_multi_instance = 1,      ///< The factory can be used to
                                            ///< create more than one component
                                            ///< at the same time
        factory_instance_count_is_size = 2  ///< The component count will be
                                            ///< interpreted as the component
                                            ///< size instead
    };
}}

///////////////////////////////////////////////////////////////////////////////
#define HPX_DEFINE_GET_COMPONENT_TYPE(component)                              \
    namespace hpx { namespace traits                                          \
    {                                                                         \
        template <> HPX_ALWAYS_EXPORT                                         \
        components::component_type component_type_database<component>::get()  \
            { return value; }                                                 \
        template <> HPX_ALWAYS_EXPORT                                         \
        void component_type_database<component>::set(                         \
            components::component_type t) { value = t; }                      \
    }}                                                                        \
/**/

#define HPX_DEFINE_GET_COMPONENT_TYPE_TEMPLATE(template_, component)          \
    namespace hpx { namespace traits                                          \
    {                                                                         \
        HPX_UTIL_STRIP(template_)                                             \
        struct component_type_database<HPX_UTIL_STRIP(component) >            \
        {                                                                     \
            static components::component_type value;                          \
                                                                              \
            HPX_ALWAYS_EXPORT static components::component_type get()         \
                { return value; }                                             \
            HPX_ALWAYS_EXPORT static void set(components::component_type t)   \
                { value = t; }                                                \
        };                                                                    \
                                                                              \
        HPX_UTIL_STRIP(template_) components::component_type                  \
        component_type_database<HPX_UTIL_STRIP(component) >::value =          \
            components::component_invalid;                                    \
    }}                                                                        \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(component, type)                 \
    namespace hpx { namespace traits                                          \
    {                                                                         \
        template <> HPX_ALWAYS_EXPORT                                         \
        components::component_type component_type_database<component>::get()  \
            { return type; }                                                  \
        template <> HPX_ALWAYS_EXPORT                                         \
        void component_type_database<component>::set(components::component_type) \
            { BOOST_ASSERT(false); }                                          \
    }}                                                                        \
/**/

#endif


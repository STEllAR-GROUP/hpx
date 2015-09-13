//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENT_COMPONENT_TYPE_MAR_26_2008_1058AM)
#define HPX_COMPONENT_COMPONENT_TYPE_MAR_26_2008_1058AM

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/traits/component_type_database.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>
#if defined(HPX_HAVE_SECURITY)
#include <hpx/components/security/capability.hpp>
#endif

#include <boost/lexical_cast.hpp>
#include <boost/cstdint.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
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
        if ((lhs_base == component_base_lco && rhs_base == component_base_lco_with_value)
            || (rhs_base == component_base_lco && lhs_base ==
                component_base_lco_with_value))
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
            { HPX_ASSERT(false); }                                          \
    }}                                                                        \
/**/

#endif


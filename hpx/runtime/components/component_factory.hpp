//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/components/component_factory.hpp

#pragma once

#ifdef DOXYGEN
/// \def HPX_REGISTER_COMPONENT(type, name, mode)
///
/// \brief Define a component factory for a component type
///
/// This macro is used create and to register a minimal component factory for
/// a component type which allows it to be remotely created using the
/// \a hpx::new_<> function.
///
/// This macro can be invoked with one, two or three arguments
///
/// \param type The \a type parameter is a (fully decorated) type of the
///             component type for which a factory should be defined.
///
/// \param name The \a name parameter specifies the name to use to register
///             the factory. This should uniquely (system-wide) identify the
///             component type. The \a name parameter must conform to the C++
///             identifier rules (without any namespace).
///             If this parameter is not given, the first parameter is used.
///
/// \param mode The \a mode parameter has to be one of the defined enumeration
///             values of the enumeration \a hpx::components::factory_state_enum.
///             The default for this parameter is
///             \a hpx::components::factory_enabled.
///
#define HPX_REGISTER_COMPONENT(type, name, mode)

#else

#include <hpx/config.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/expand.hpp>
#include <hpx/preprocessor/nargs.hpp>
#include <hpx/runtime/components/component_factory_base.hpp>
#include <hpx/runtime/components/component_registry.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

///////////////////////////////////////////////////////////////////////////////
// This macro is used create and to register a minimal component factory with
// Hpx.Plugin.
#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(...)                           \
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_(__VA_ARGS__)                      \
/**/

#define HPX_REGISTER_COMPONENT(...)                                           \
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_(__VA_ARGS__)                      \
/**/

#define HPX_REGISTER_ENABLED_COMPONENT_FACTORY(ComponentType, componentname)  \
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_3(                                 \
        ComponentType, componentname, ::hpx::components::factory_enabled)     \
    HPX_DEFINE_GET_COMPONENT_TYPE(ComponentType::wrapped_type)                \
/**/

#define HPX_REGISTER_DISABLED_COMPONENT_FACTORY(ComponentType, componentname) \
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_3(                                 \
        ComponentType, componentname, ::hpx::components::factory_disabled)    \
    HPX_DEFINE_GET_COMPONENT_TYPE(ComponentType::wrapped_type)                \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_(...)                          \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                 \
        HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_, HPX_PP_NARGS(__VA_ARGS__)    \
    )(__VA_ARGS__))                                                           \
/**/
#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_1(ComponentType)               \
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_3(                                 \
        ComponentType, ComponentType, ::hpx::components::factory_check)       \
    HPX_DEFINE_GET_COMPONENT_TYPE(ComponentType::wrapped_type)                \
/**/
#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_2(ComponentType, componentname)\
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_3(                                 \
        ComponentType, componentname, ::hpx::components::factory_check)       \
    HPX_DEFINE_GET_COMPONENT_TYPE(ComponentType::wrapped_type)                \
/**/
#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_3(                             \
        ComponentType, componentname, state)                                  \
    HPX_REGISTER_COMPONENT_HEAP(ComponentType)                                \
    HPX_REGISTER_COMPONENT_FACTORY(componentname)                             \
    HPX_DEFINE_COMPONENT_NAME(ComponentType::type_holder, componentname)      \
    HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_3(                                \
        ComponentType, componentname, state)                                  \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC(...)                   \
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC_(__VA_ARGS__)              \
/**/

// same as above, just a better name

/// This macro is used create and to register a minimal component factory for
/// a component type which allows it to be remotely created using the
/// hpx::new_<> function.
/// This macro can be invoked with one, two or three arguments
#define HPX_REGISTER_COMPONENT_DYNAMIC(...)                                   \
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC_(__VA_ARGS__)              \
/**/

#define HPX_REGISTER_ENABLED_COMPONENT_FACTORY_DYNAMIC(ComponentType,         \
            componentname)                                                    \
        HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC_3(                     \
            ComponentType, componentname, ::hpx::components::factory_enabled) \
        HPX_DEFINE_GET_COMPONENT_TYPE(ComponentType::wrapped_type)            \
/**/

#define HPX_REGISTER_DISABLED_COMPONENT_FACTORY_DYNAMIC(ComponentType,        \
            componentname)                                                    \
        HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC_3(                     \
            ComponentType, componentname, ::hpx::components::factory_disabled)\
        HPX_DEFINE_GET_COMPONENT_TYPE(ComponentType::wrapped_type)            \
/**/


#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC_(...)                  \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                 \
        HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC_,                      \
            HPX_PP_NARGS(__VA_ARGS__)                                         \
    )(__VA_ARGS__))                                                           \
/**/
#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC_1(ComponentType)       \
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC_3(                         \
        ComponentType, ComponentType, ::hpx::components::factory_check)       \
    HPX_DEFINE_GET_COMPONENT_TYPE(ComponentType::wrapped_type)                \
/**/
#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC_2(ComponentType,       \
        componentname)                                                        \
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC_3(                         \
        ComponentType, componentname, ::hpx::components::factory_check)       \
    HPX_DEFINE_GET_COMPONENT_TYPE(ComponentType::wrapped_type)                \
/**/
#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC_3(                     \
        ComponentType, componentname, state)                                  \
    HPX_REGISTER_COMPONENT_HEAP(ComponentType)                                \
    HPX_DEFINE_COMPONENT_NAME(ComponentType::type_holder, componentname)      \
    HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_DYNAMIC_3(                        \
        ComponentType, componentname, state)                                  \
/**/

#else    // COMPUTE DEVICE CODE

#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(...) /**/

#define HPX_REGISTER_COMPONENT(...) /**/

#define HPX_REGISTER_ENABLED_COMPONENT_FACTORY(ComponentType, componentname)
/**/
#define HPX_REGISTER_DISABLED_COMPONENT_FACTORY(ComponentType, componentname)
/**/
///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(...)         /**/
///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC(...) /**/

#define HPX_REGISTER_COMPONENT_DYNAMIC(...) /**/

#define HPX_REGISTER_ENABLED_COMPONENT_FACTORY_DYNAMIC(                        \
    ComponentType, componentname) /**/

#define HPX_REGISTER_DISABLED_COMPONENT_FACTORY_DYNAMIC(                       \
    ComponentType, componentname) /**/

#endif
#endif

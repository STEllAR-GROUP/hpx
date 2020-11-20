//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2017      Thomas Heller
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/expand.hpp>
#include <hpx/preprocessor/nargs.hpp>
#include <hpx/preprocessor/strip_parens.hpp>

///////////////////////////////////////////////////////////////////////////////
/// This macro is used create and to register a minimal component factory with
/// Hpx.Plugin. This macro may be used if the registered component factory is
/// the only factory to be exposed from a particular module. If more than one
/// factory needs to be exposed the \a HPX_REGISTER_COMPONENT_FACTORY and
/// \a HPX_REGISTER_COMPONENT_MODULE macros should be used instead.
#define HPX_REGISTER_DERIVED_COMPONENT_FACTORY(...)                           \
    HPX_REGISTER_DERIVED_COMPONENT_FACTORY_(__VA_ARGS__)                      \
/**/

#define HPX_REGISTER_DERIVED_COMPONENT_FACTORY_(...)                          \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                 \
        HPX_REGISTER_DERIVED_COMPONENT_FACTORY_, HPX_PP_NARGS(__VA_ARGS__)    \
    )(__VA_ARGS__))                                                           \
/**/
#define HPX_REGISTER_DERIVED_COMPONENT_FACTORY_3(ComponentType, componentname,\
        basecomponentname)                                                    \
    HPX_REGISTER_DERIVED_COMPONENT_FACTORY_4(                                 \
        ComponentType, componentname, basecomponentname,                      \
        ::hpx::components::factory_check)                                     \
    HPX_DEFINE_GET_COMPONENT_TYPE(ComponentType::wrapped_type)                \
/**/
#define HPX_REGISTER_DERIVED_COMPONENT_FACTORY_4(ComponentType,               \
        componentname, basecomponentname, state)                              \
    HPX_REGISTER_COMPONENT_HEAP(ComponentType)                                \
    HPX_REGISTER_COMPONENT_FACTORY(componentname)                             \
    HPX_DEFINE_COMPONENT_NAME(ComponentType::type_holder, componentname,      \
        basecomponentname)                                                    \
    HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_3(ComponentType,                  \
        componentname, state)                                                 \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_DERIVED_COMPONENT_FACTORY_DYNAMIC(...)                   \
    HPX_REGISTER_DERIVED_COMPONENT_FACTORY_DYNAMIC_(__VA_ARGS__)              \
/**/

#define HPX_REGISTER_DERIVED_COMPONENT_FACTORY_DYNAMIC_(...)                  \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                 \
        HPX_REGISTER_DERIVED_COMPONENT_FACTORY_DYNAMIC_,                      \
            HPX_PP_NARGS(__VA_ARGS__)                                         \
    )(__VA_ARGS__))                                                           \
/**/
#define HPX_REGISTER_DERIVED_COMPONENT_FACTORY_DYNAMIC_3(ComponentType,       \
        componentname, basecomponentname)                                     \
    HPX_REGISTER_DERIVED_COMPONENT_FACTORY_DYNAMIC_4(                         \
        ComponentType, componentname, basecomponentname,                      \
        ::hpx::components::factory_check)                                     \
    HPX_DEFINE_GET_COMPONENT_TYPE(ComponentType::wrapped_type)                \
/**/
#define HPX_REGISTER_DERIVED_COMPONENT_FACTORY_DYNAMIC_4(ComponentType,       \
        componentname, basecomponentname, state)                              \
    HPX_REGISTER_COMPONENT_HEAP(ComponentType)                                \
    HPX_DEFINE_COMPONENT_NAME(ComponentType::type_holder, componentname,      \
        basecomponentname)                                                    \
    HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_DYNAMIC_3(ComponentType,          \
        componentname, state)                                                 \
/**/
#else    // COMPUTE DEVICE CODE

#define HPX_REGISTER_DERIVED_COMPONENT_FACTORY(...) /**/

#define HPX_REGISTER_DERIVED_COMPONENT_FACTORY_(...) /**/
#define HPX_REGISTER_DERIVED_COMPONENT_FACTORY_3(                              \
    ComponentType, componentname, basecomponentname) /**/
#define HPX_REGISTER_DERIVED_COMPONENT_FACTORY_4(                              \
    ComponentType, componentname, basecomponentname, state) /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_DERIVED_COMPONENT_FACTORY_DYNAMIC(...) /**/

#define HPX_REGISTER_DERIVED_COMPONENT_FACTORY_DYNAMIC_(...) /**/
#define HPX_REGISTER_DERIVED_COMPONENT_FACTORY_DYNAMIC_3(                      \
    ComponentType, componentname, basecomponentname) /**/
#define HPX_REGISTER_DERIVED_COMPONENT_FACTORY_DYNAMIC_4(                      \
    ComponentType, componentname, basecomponentname, state) /**/

#endif

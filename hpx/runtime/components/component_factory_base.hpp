//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/runtime_configuration/component_registry_base.hpp>

///////////////////////////////////////////////////////////////////////////////
/// This macro is used to register the given component factory with
/// Hpx.Plugin. This macro has to be used for each of the component factories.
#define HPX_REGISTER_COMPONENT_FACTORY(componentname)                         \
    HPX_INIT_REGISTRY_FACTORY_STATIC(HPX_PLUGIN_COMPONENT_PREFIX,             \
        componentname, factory)                                               \
/**/

///////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_APPLICATION_NAME) && !defined(HPX_HAVE_STATIC_LINKING)
/// This macro is used to define the required Hpx.Plugin entry points. This
/// macro has to be used in exactly one compilation unit of a component module.
#define HPX_REGISTER_COMPONENT_MODULE()                                       \
    HPX_PLUGIN_EXPORT_LIST(HPX_PLUGIN_COMPONENT_PREFIX, factory)              \
    HPX_REGISTER_REGISTRY_MODULE()                                            \
/**/
#define HPX_REGISTER_COMPONENT_MODULE_DYNAMIC()                               \
    HPX_PLUGIN_EXPORT_LIST_DYNAMIC(HPX_PLUGIN_COMPONENT_PREFIX, factory)      \
    HPX_REGISTER_REGISTRY_MODULE_DYNAMIC()                                    \
/**/
#else
// in executables (when HPX_APPLICATION_NAME is defined) this needs to expand
// to nothing
#if defined(HPX_HAVE_STATIC_LINKING)
#define HPX_REGISTER_COMPONENT_MODULE()                                       \
    HPX_PLUGIN_EXPORT_LIST(HPX_PLUGIN_COMPONENT_PREFIX, factory)              \
    HPX_REGISTER_REGISTRY_MODULE()                                            \
/**/
#else
#define HPX_REGISTER_COMPONENT_MODULE()
#endif
#define HPX_REGISTER_COMPONENT_MODULE_DYNAMIC()
#endif

#else    // COMPUTE DEVICE CODE

#define HPX_REGISTER_COMPONENT_FACTORY(componentname) /**/
#define HPX_REGISTER_COMPONENT_MODULE()               /**/
#define HPX_REGISTER_COMPONENT_MODULE_DYNAMIC()       /**/

#endif

//  Copyright (c) 2007-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file component_commandline.hpp
/// \page HPX_REGISTER_COMPONENT_MODULE
/// \headerfile hpx/components.hpp

// Make HPX inspect tool happy: hpxinspect:nounnamed

#pragma once

#include <hpx/config.hpp>

#include <hpx/modules/plugin.hpp>
#include <hpx/modules/preprocessor.hpp>

#include <map>
#include <string>

////////////////////////////////////////////////////////////////////////////////
// from hpx/runtime_configuration/component_command_line_base.hpp

/// The macro \a HPX_REGISTER_COMMANDLINE_REGISTRY is used to register the given
/// component factory with Hpx.Plugin. This macro has to be used for each of the
/// components.
#define HPX_REGISTER_COMMANDLINE_REGISTRY(RegistryType, componentname)         \
    HPX_PLUGIN_EXPORT(HPX_PLUGIN_COMPONENT_PREFIX,                             \
        hpx::components::component_commandline_base, RegistryType,             \
        componentname, commandline_options)                                    \
/**/
#define HPX_REGISTER_COMMANDLINE_REGISTRY_DYNAMIC(RegistryType, componentname) \
    HPX_PLUGIN_EXPORT_DYNAMIC(HPX_PLUGIN_COMPONENT_PREFIX,                     \
        hpx::components::component_commandline_base, RegistryType,             \
        componentname, commandline_options)                                    \
/**/

/// The macro \a HPX_REGISTER_COMMANDLINE_OPTIONS is used to define the required
/// Hpx.Plugin entry point for the command line option registry. This macro has
/// to be used in not more than one compilation unit of a component module.
#define HPX_REGISTER_COMMANDLINE_OPTIONS()                                     \
    HPX_PLUGIN_EXPORT_LIST(HPX_PLUGIN_COMPONENT_PREFIX, commandline_options)   \
    HPX_INIT_REGISTRY_COMMANDLINE_STATIC(                                      \
        HPX_PLUGIN_COMPONENT_PREFIX, commandline_options)                      \
/**/
#define HPX_REGISTER_COMMANDLINE_OPTIONS_DYNAMIC()                             \
    HPX_PLUGIN_EXPORT_LIST_DYNAMIC(                                            \
        HPX_PLUGIN_COMPONENT_PREFIX, commandline_options)                      \
/**/

////////////////////////////////////////////////////////////////////////////////
// from hpx/runtime_configuration/component_factory_base.hpp

#if !defined(HPX_COMPUTE_DEVICE_CODE)

/// This macro is used to register the given component factory with Hpx.Plugin.
/// This macro has to be used for each of the component factories.
#define HPX_REGISTER_COMPONENT_FACTORY(componentname)                          \
    HPX_INIT_REGISTRY_FACTORY_STATIC(                                          \
        HPX_PLUGIN_COMPONENT_PREFIX, componentname, factory)                   \
/**/

////////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_APPLICATION_NAME) && !defined(HPX_HAVE_STATIC_LINKING)

/// This macro is used to define the required Hpx.Plugin entry points. This
/// macro has to be used in exactly one compilation unit of a component module.
#define HPX_REGISTER_COMPONENT_MODULE()                                        \
    HPX_PLUGIN_EXPORT_LIST(HPX_PLUGIN_COMPONENT_PREFIX, factory)               \
    HPX_REGISTER_REGISTRY_MODULE()                                             \
/**/
#define HPX_REGISTER_COMPONENT_MODULE_DYNAMIC()                                \
    HPX_PLUGIN_EXPORT_LIST_DYNAMIC(HPX_PLUGIN_COMPONENT_PREFIX, factory)       \
    HPX_REGISTER_REGISTRY_MODULE_DYNAMIC()                                     \
    /**/

#else

// in executables (when HPX_APPLICATION_NAME is defined) this needs to expand to
// nothing
#if defined(HPX_HAVE_STATIC_LINKING)
#define HPX_REGISTER_COMPONENT_MODULE()                                        \
    HPX_PLUGIN_EXPORT_LIST(HPX_PLUGIN_COMPONENT_PREFIX, factory)               \
    HPX_REGISTER_REGISTRY_MODULE()                                             \
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

////////////////////////////////////////////////////////////////////////////////
// from hpx/runtime_configuration/component_registry_base.hpp

/// This macro is used to register the given component factory with Hpx.Plugin.
/// This macro has to be used for each of the components.
#define HPX_REGISTER_COMPONENT_REGISTRY(RegistryType, componentname)           \
    HPX_PLUGIN_EXPORT(HPX_PLUGIN_COMPONENT_PREFIX,                             \
        hpx::components::component_registry_base, RegistryType, componentname, \
        registry)                                                              \
/**/
#define HPX_REGISTER_COMPONENT_REGISTRY_DYNAMIC(RegistryType, componentname)   \
    HPX_PLUGIN_EXPORT_DYNAMIC(HPX_PLUGIN_COMPONENT_PREFIX,                     \
        hpx::components::component_registry_base, RegistryType, componentname, \
        registry)                                                              \
/**/

///////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_APPLICATION_NAME)

/// This macro is used to define the required Hpx.Plugin entry points. This
/// macro has to be used in exactly one compilation unit of a component module.
#define HPX_REGISTER_REGISTRY_MODULE()                                         \
    HPX_PLUGIN_EXPORT_LIST(HPX_PLUGIN_COMPONENT_PREFIX, registry)              \
    HPX_INIT_REGISTRY_MODULE_STATIC(HPX_PLUGIN_COMPONENT_PREFIX, registry)     \
/**/
#define HPX_REGISTER_REGISTRY_MODULE_DYNAMIC()                                 \
    HPX_PLUGIN_EXPORT_LIST_DYNAMIC(HPX_PLUGIN_COMPONENT_PREFIX, registry)      \
    /**/

#else

// in executables (when HPX_APPLICATION_NAME is defined) this needs to expand
// to nothing
#define HPX_REGISTER_REGISTRY_MODULE()
#define HPX_REGISTER_REGISTRY_MODULE_DYNAMIC()

#endif

////////////////////////////////////////////////////////////////////////////////
// from hpx/runtime_configuration/plugin_registry_base.hpp

/// This macro is used to register the given component factory with Hpx.Plugin.
/// This macro has to be used for each of the components.
#define HPX_REGISTER_PLUGIN_BASE_REGISTRY(PluginType, name)                    \
    HPX_PLUGIN_EXPORT(HPX_PLUGIN_PLUGIN_PREFIX,                                \
        hpx::plugins::plugin_registry_base, PluginType, name, plugin)          \
    /**/

/// This macro is used to define the required Hpx.Plugin entry points. This
/// macro has to be used in exactly one compilation unit of a component module.
#define HPX_REGISTER_PLUGIN_REGISTRY_MODULE()                                  \
    HPX_PLUGIN_EXPORT_LIST(HPX_PLUGIN_PLUGIN_PREFIX, plugin)                   \
    /**/

#define HPX_REGISTER_PLUGIN_REGISTRY_MODULE_DYNAMIC()                          \
    HPX_PLUGIN_EXPORT_LIST_DYNAMIC(HPX_PLUGIN_PLUGIN_PREFIX, plugin)           \
    /**/

////////////////////////////////////////////////////////////////////////////////
// from hpx/runtime_configuration/static_factory_data.hpp

#define HPX_DECLARE_FACTORY_STATIC(name, base)                                 \
    extern "C" HPX_PLUGIN_EXPORT_API std::map<std::string, hpx::any_nonser>*   \
        HPX_PLUGIN_API                                                         \
        HPX_PLUGIN_LIST_NAME(name, base)() /**/

#define HPX_DEFINE_FACTORY_STATIC(module, name, base)                          \
    {HPX_PP_STRINGIZE(module), HPX_PLUGIN_LIST_NAME(name, base)} /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_INIT_REGISTRY_MODULE_STATIC(name, base)                            \
    HPX_DECLARE_FACTORY_STATIC(name, base);                                    \
    namespace {                                                                \
        struct HPX_PP_CAT(init_registry_module_static_, name)                  \
        {                                                                      \
            HPX_PP_CAT(init_registry_module_static_, name)()                   \
            {                                                                  \
                hpx::components::static_factory_load_data_type data =          \
                    HPX_DEFINE_FACTORY_STATIC(HPX_COMPONENT_NAME, name, base); \
                hpx::components::init_registry_module(data);                   \
            }                                                                  \
        } HPX_PP_CAT(module_data_, __LINE__);                                  \
    }                                                                          \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_INIT_REGISTRY_FACTORY_STATIC(name, componentname, base)            \
    HPX_DECLARE_FACTORY_STATIC(name, base);                                    \
    namespace {                                                                \
        struct HPX_PP_CAT(init_registry_factory_static_, componentname)        \
        {                                                                      \
            HPX_PP_CAT(init_registry_factory_static_, componentname)()         \
            {                                                                  \
                hpx::components::static_factory_load_data_type data =          \
                    HPX_DEFINE_FACTORY_STATIC(componentname, name, base);      \
                hpx::components::init_registry_factory(data);                  \
            }                                                                  \
        } HPX_PP_CAT(componentname, HPX_PP_CAT(_factory_data_, __LINE__));     \
    }                                                                          \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_INIT_REGISTRY_COMMANDLINE_STATIC(name, base)                       \
    HPX_DECLARE_FACTORY_STATIC(name, base);                                    \
    namespace {                                                                \
        struct HPX_PP_CAT(init_registry_module_commandline_, name)             \
        {                                                                      \
            HPX_PP_CAT(init_registry_module_commandline_, name)()              \
            {                                                                  \
                hpx::components::static_factory_load_data_type data =          \
                    HPX_DEFINE_FACTORY_STATIC(HPX_COMPONENT_NAME, name, base); \
                hpx::components::init_registry_commandline(data);              \
            }                                                                  \
        } HPX_PP_CAT(module_commandline_data_, __LINE__);                      \
    }                                                                          \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_INIT_REGISTRY_STARTUP_SHUTDOWN_STATIC(name, base)                  \
    HPX_DECLARE_FACTORY_STATIC(name, base);                                    \
    namespace {                                                                \
        struct HPX_PP_CAT(init_registry_module_startup_shutdown_, name)        \
        {                                                                      \
            HPX_PP_CAT(init_registry_module_startup_shutdown_, name)()         \
            {                                                                  \
                hpx::components::static_factory_load_data_type data =          \
                    HPX_DEFINE_FACTORY_STATIC(HPX_COMPONENT_NAME, name, base); \
                hpx::components::init_registry_startup_shutdown(data);         \
            }                                                                  \
        } HPX_PP_CAT(module_startup_shutdown_data_, __LINE__);                 \
    }                                                                          \
    /**/

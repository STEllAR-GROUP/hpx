//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/plugin.hpp>
#include <hpx/runtime_local/shutdown_function.hpp>
#include <hpx/runtime_local/startup_function.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components {

    ///////////////////////////////////////////////////////////////////////////
    /// The \a component_startup_shutdown_base has to be used as a base class
    /// for all component startup/shutdown registries.
    struct component_startup_shutdown_base
    {
        virtual ~component_startup_shutdown_base() = default;

        /// \brief Return any startup function for this component
        ///
        /// \param startup  [in, out] The module is expected to fill this
        ///                 function object with a reference to a startup
        ///                 function. This function will be executed by the
        ///                 runtime system during system startup.
        ///
        /// \return Returns \a true if the parameter \a startup has been
        ///         successfully initialized with the startup function.
        virtual bool get_startup_function(
            startup_function_type& startup, bool& pre_startup) = 0;

        /// \brief Return any startup function for this component
        ///
        /// \param shutdown  [in, out] The module is expected to fill this
        ///                 function object with a reference to a startup
        ///                 function. This function will be executed by the
        ///                 runtime system during system startup.
        ///
        /// \return Returns \a true if the parameter \a shutdown has been
        ///         successfully initialized with the shutdown function.
        virtual bool get_shutdown_function(
            shutdown_function_type& shutdown, bool& pre_shutdown) = 0;
    };
}}    // namespace hpx::components

///////////////////////////////////////////////////////////////////////////////
/// This macro is used to register the given component factory with
/// Hpx.Plugin. This macro has to be used for each of the components.
#define HPX_REGISTER_STARTUP_SHUTDOWN_REGISTRY(RegistryType, componentname)    \
    HPX_PLUGIN_EXPORT(HPX_PLUGIN_COMPONENT_PREFIX,                             \
        hpx::components::component_startup_shutdown_base, RegistryType,        \
        componentname, startup_shutdown)                                       \
/**/
#define HPX_REGISTER_STARTUP_SHUTDOWN_REGISTRY_DYNAMIC(                        \
    RegistryType, componentname)                                               \
    HPX_PLUGIN_EXPORT_DYNAMIC(HPX_PLUGIN_COMPONENT_PREFIX,                     \
        hpx::components::component_startup_shutdown_base, RegistryType,        \
        componentname, startup_shutdown)                                       \
/**/

/// This macro is used to define the required Hpx.Plugin entry point for the
/// startup/shutdown registry. This macro has to be used in not more than one
/// compilation unit of a component module.
#define HPX_REGISTER_STARTUP_SHUTDOWN_FUNCTIONS()                              \
    HPX_PLUGIN_EXPORT_LIST(HPX_PLUGIN_COMPONENT_PREFIX, startup_shutdown)      \
    HPX_INIT_REGISTRY_STARTUP_SHUTDOWN_STATIC(                                 \
        HPX_PLUGIN_COMPONENT_PREFIX, startup_shutdown)                         \
/**/
#define HPX_REGISTER_STARTUP_SHUTDOWN_FUNCTIONS_DYNAMIC()                      \
    HPX_PLUGIN_EXPORT_LIST_DYNAMIC(                                            \
        HPX_PLUGIN_COMPONENT_PREFIX, startup_shutdown)                         \
    /**/

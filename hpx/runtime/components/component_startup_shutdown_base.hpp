//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENT_STARTUP_SHUTDOWN_BASE_SEP_20_2011_0809PM)
#define HPX_COMPONENT_STARTUP_SHUTDOWN_BASE_SEP_20_2011_0809PM

#include <hpx/config.hpp>
#include <hpx/runtime/shutdown_function.hpp>
#include <hpx/runtime/startup_function.hpp>
#include <hpx/util/plugin.hpp>
#include <hpx/util/plugin/export_plugin.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a component_startup_shutdown_base has to be used as a base class
    /// for all component startup/shutdown registries.
    struct HPX_EXPORT component_startup_shutdown_base
    {
        virtual ~component_startup_shutdown_base() {}

        /// \brief Return any startup function for this component
        ///
        /// \param startup  [in, out] The module is expected to fill this
        ///                 function object with a reference to a startup
        ///                 function. This function will be executed by the
        ///                 runtime system during system startup.
        ///
        /// \return Returns \a true if the parameter \a startup has been
        ///         successfully initialized with the startup function.
        virtual bool get_startup_function(startup_function_type& startup,
            bool& pre_startup) = 0;

        /// \brief Return any startup function for this component
        ///
        /// \param shutdown  [in, out] The module is expected to fill this
        ///                 function object with a reference to a startup
        ///                 function. This function will be executed by the
        ///                 runtime system during system startup.
        ///
        /// \return Returns \a true if the parameter \a shutdown has been
        ///         successfully initialized with the shutdown function.
        virtual bool get_shutdown_function(shutdown_function_type& shutdown,
            bool& pre_shutdown) = 0;
    };
}}

///////////////////////////////////////////////////////////////////////////////
/// This macro is used to register the given component factory with
/// Hpx.Plugin. This macro has to be used for each of the components.
#define HPX_REGISTER_STARTUP_SHUTDOWN_REGISTRY(RegistryType, componentname)   \
    HPX_PLUGIN_EXPORT(HPX_PLUGIN_COMPONENT_PREFIX,                            \
        hpx::components::component_startup_shutdown_base, RegistryType,       \
        componentname, startup_shutdown)                                      \
/**/
#define HPX_REGISTER_STARTUP_SHUTDOWN_REGISTRY_DYNAMIC(RegistryType,          \
        componentname)                                                        \
    HPX_PLUGIN_EXPORT_DYNAMIC(HPX_PLUGIN_COMPONENT_PREFIX,                    \
        hpx::components::component_startup_shutdown_base, RegistryType,       \
        componentname, startup_shutdown)                                      \
/**/

/// This macro is used to define the required Hpx.Plugin entry point for the
/// startup/shutdown registry. This macro has to be used in not more than one
/// compilation unit of a component module.
#define HPX_REGISTER_STARTUP_SHUTDOWN_FUNCTIONS()                             \
    HPX_PLUGIN_EXPORT_LIST(HPX_PLUGIN_COMPONENT_PREFIX, startup_shutdown);    \
    HPX_INIT_REGISTRY_STARTUP_SHUTDOWN_STATIC(HPX_PLUGIN_COMPONENT_PREFIX,    \
        startup_shutdown)                                                     \
/**/
#define HPX_REGISTER_STARTUP_SHUTDOWN_FUNCTIONS_DYNAMIC()                     \
    HPX_PLUGIN_EXPORT_LIST_DYNAMIC(HPX_PLUGIN_COMPONENT_PREFIX,               \
        startup_shutdown)                                                     \
/**/

#endif


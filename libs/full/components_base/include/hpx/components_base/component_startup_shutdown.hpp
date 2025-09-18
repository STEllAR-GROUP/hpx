//  Copyright (c) 2007-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file component_commandline.hpp
/// \page HPX_REGISTER_STARTUP_MODULE
/// \headerfile hpx/components.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/preprocessor.hpp>
#include <hpx/runtime_local/component_startup_shutdown_base.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::components {

    ///////////////////////////////////////////////////////////////////////////
    /// The \a component_startup_shutdown class provides a minimal
    /// implementation of a component's startup/shutdown function provider.
    template <bool (*Startup)(startup_function_type&, bool&),
        bool (*Shutdown)(shutdown_function_type&, bool&)>
    struct component_startup_shutdown : component_startup_shutdown_base
    {
        /// \brief Return any startup function for this component
        ///
        /// \param startup  [in, out] The module is expected to fill this
        ///                 function object with a reference to a startup
        ///                 function. This function will be executed by the
        ///                 runtime system during system startup.
        /// \param pre_startup
        ///
        /// \return Returns \a true if the parameter \a startup has been
        ///         successfully initialized with the startup function.
        bool get_startup_function(
            startup_function_type& startup, bool& pre_startup) override
        {
            return Startup(startup, pre_startup);
        }

        /// \brief Return any startup function for this component
        ///
        /// \param shutdown [in, out] The module is expected to fill this
        ///                 function object with a reference to a startup
        ///                 function. This function will be executed by the
        ///                 runtime system during system startup.
        /// \param pre_shutdown
        ///
        /// \return Returns \a true if the parameter \a shutdown has been
        ///         successfully initialized with the shutdown function.
        bool get_shutdown_function(
            shutdown_function_type& shutdown, bool& pre_shutdown) override
        {
            return Shutdown(shutdown, pre_shutdown);
        }
    };
}    // namespace hpx::components

///////////////////////////////////////////////////////////////////////////////
#define HPX_DEFINE_COMPONENT_STARTUP_SHUTDOWN(startup_, shutdown_)             \
    namespace hpx::components::startup_shutdown_provider {                     \
        bool HPX_PP_CAT(HPX_PLUGIN_COMPONENT_PREFIX, _startup)(                \
            startup_function_type & startup_func, bool& pre_startup)           \
        {                                                                      \
            hpx::function<bool(startup_function_type&, bool&)> tmp =           \
                static_cast<bool (*)(startup_function_type&, bool&)>(          \
                    startup_);                                                 \
            if (!!tmp)                                                         \
            {                                                                  \
                return tmp(startup_func, pre_startup);                         \
            }                                                                  \
            return false;                                                      \
        }                                                                      \
        bool HPX_PP_CAT(HPX_PLUGIN_COMPONENT_PREFIX, _shutdown)(               \
            shutdown_function_type & shutdown_func, bool& pre_shutdown)        \
        {                                                                      \
            hpx::function<bool(shutdown_function_type&, bool&)> tmp =          \
                static_cast<bool (*)(shutdown_function_type&, bool&)>(         \
                    shutdown_);                                                \
            if (!!tmp)                                                         \
            {                                                                  \
                return tmp(shutdown_func, pre_shutdown);                       \
            }                                                                  \
            return false;                                                      \
        }                                                                      \
    }                                                                          \
    /***/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_STARTUP_SHUTDOWN_MODULE_(startup, shutdown)               \
    HPX_DEFINE_COMPONENT_STARTUP_SHUTDOWN(startup, shutdown)                   \
    namespace hpx::components::startup_shutdown_provider {                     \
        using HPX_PP_CAT(HPX_PLUGIN_COMPONENT_PREFIX, _provider) =             \
            component_startup_shutdown<HPX_PP_CAT(HPX_PLUGIN_COMPONENT_PREFIX, \
                                           _startup),                          \
                HPX_PP_CAT(HPX_PLUGIN_COMPONENT_PREFIX, _shutdown)>;           \
    }                                                                          \
    namespace hpx::components {                                                \
        template struct component_startup_shutdown<                            \
            startup_shutdown_provider::HPX_PP_CAT(                             \
                HPX_PLUGIN_COMPONENT_PREFIX, _startup),                        \
            startup_shutdown_provider::HPX_PP_CAT(                             \
                HPX_PLUGIN_COMPONENT_PREFIX, _shutdown)>;                      \
    }                                                                          \
    /**/

#define HPX_REGISTER_STARTUP_SHUTDOWN_MODULE(startup, shutdown)                \
    HPX_REGISTER_STARTUP_SHUTDOWN_FUNCTIONS()                                  \
    HPX_REGISTER_STARTUP_SHUTDOWN_MODULE_(startup, shutdown)                   \
    HPX_REGISTER_STARTUP_SHUTDOWN_REGISTRY(                                    \
        hpx::components::startup_shutdown_provider::HPX_PP_CAT(                \
            HPX_PLUGIN_COMPONENT_PREFIX, _provider),                           \
        startup_shutdown)                                                      \
    /**/
#define HPX_REGISTER_STARTUP_SHUTDOWN_MODULE_DYNAMIC(startup, shutdown)        \
    HPX_REGISTER_STARTUP_SHUTDOWN_FUNCTIONS_DYNAMIC()                          \
    HPX_REGISTER_STARTUP_SHUTDOWN_MODULE_(startup, shutdown)                   \
    HPX_REGISTER_STARTUP_SHUTDOWN_REGISTRY_DYNAMIC(                            \
        hpx::components::startup_shutdown_provider::HPX_PP_CAT(                \
            HPX_PLUGIN_COMPONENT_PREFIX, _provider),                           \
        startup_shutdown)                                                      \
    /**/

/**
 * @brief Macro to register a startup module with the HPX runtime.
 *
 * This macro facilitates the registration of a startup module with the HPX
 * runtime system. A startup module typically contains initialization code
 * that should be executed when the HPX runtime starts.
 *
 * @param startup The name of the startup function to be registered.
 */
#define HPX_REGISTER_STARTUP_MODULE(startup)                                   \
    HPX_REGISTER_STARTUP_SHUTDOWN_FUNCTIONS()                                  \
    HPX_REGISTER_STARTUP_SHUTDOWN_MODULE_(startup, 0)                          \
    HPX_REGISTER_STARTUP_SHUTDOWN_REGISTRY(                                    \
        hpx::components::startup_shutdown_provider::HPX_PP_CAT(                \
            HPX_PLUGIN_COMPONENT_PREFIX, _provider),                           \
        startup_shutdown)                                                      \
    /**/
#define HPX_REGISTER_STARTUP_MODULE_DYNAMIC(startup)                           \
    HPX_REGISTER_STARTUP_SHUTDOWN_FUNCTIONS_DYNAMIC()                          \
    HPX_REGISTER_STARTUP_SHUTDOWN_MODULE_(startup, 0)                          \
    HPX_REGISTER_STARTUP_SHUTDOWN_REGISTRY_DYNAMIC(                            \
        hpx::components::startup_shutdown_provider::HPX_PP_CAT(                \
            HPX_PLUGIN_COMPONENT_PREFIX, _provider),                           \
        startup_shutdown)                                                      \
    /**/

#define HPX_REGISTER_SHUTDOWN_MODULE(shutdown)                                 \
    HPX_REGISTER_STARTUP_SHUTDOWN_FUNCTIONS()                                  \
    HPX_REGISTER_STARTUP_SHUTDOWN_MODULE_(0, shutdown)                         \
    HPX_REGISTER_STARTUP_SHUTDOWN_REGISTRY(                                    \
        hpx::components::startup_shutdown_provider::HPX_PP_CAT(                \
            HPX_PLUGIN_COMPONENT_PREFIX, _provider),                           \
        startup_shutdown)                                                      \
    /**/
#define HPX_REGISTER_SHUTDOWN_MODULE_DYNAMIC(shutdown)                         \
    HPX_REGISTER_STARTUP_SHUTDOWN_FUNCTIONS_DYNAMIC()                          \
    HPX_REGISTER_STARTUP_SHUTDOWN_MODULE_(0, shutdown)                         \
    HPX_REGISTER_STARTUP_SHUTDOWN_REGISTRY_DYNAMIC(                            \
        hpx::components::startup_shutdown_provider::HPX_PP_CAT(                \
            HPX_PLUGIN_COMPONENT_PREFIX, _provider),                           \
        startup_shutdown)                                                      \
    /**/

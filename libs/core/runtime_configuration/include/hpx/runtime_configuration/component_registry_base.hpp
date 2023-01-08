//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/plugin.hpp>
#include <hpx/runtime_configuration/static_factory_data.hpp>

#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::components {

    ///////////////////////////////////////////////////////////////////////////
    /// The \a component_registry_base has to be used as a base class for all
    /// component registries.
    struct HPX_CORE_EXPORT component_registry_base
    {
        virtual ~component_registry_base() = default;

        /// \brief Return the ini-information for all contained components
        ///
        /// \param fillini  [in, out] The module is expected to fill this vector
        ///                 with the ini-information (one line per vector
        ///                 element) for all components implemented in this
        ///                 module.
        ///
        /// \return Returns \a true if the parameter \a fillini has been
        ///         successfully initialized with the registry data of all
        ///         implemented in this module.
        virtual bool get_component_info(std::vector<std::string>& fillini,
            std::string const& filepath, bool is_static = false) = 0;

        /// \brief Return the unique identifier of the component type this
        ///        factory is responsible for
        ///
        /// \param locality     [in] The id of the locality this factory
        ///                     is responsible for.
        /// \param agas_client  [in] The AGAS client to use for component id
        ///                     registration (if needed).
        ///
        /// \return Returns the unique identifier of the component type this
        ///         factory instance is responsible for. This function throws on
        ///         any error.
        virtual void register_component_type() = 0;
    };
}    // namespace hpx::components

////////////////////////////////////////////////////////////////////////////////
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

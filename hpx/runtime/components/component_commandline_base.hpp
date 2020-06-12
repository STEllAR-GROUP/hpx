//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/plugin.hpp>
#include <hpx/modules/program_options.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a component_commandline_base has to be used as a base class
    /// for all component command-line line handling registries.
    struct HPX_EXPORT component_commandline_base
    {
        virtual ~component_commandline_base() {}

        /// \brief Return any additional command line options valid for this
        ///        component
        ///
        /// \return The module is expected to fill a options_description object
        ///         with any additional command line options this component
        ///         will handle.
        ///
        /// \note   This function will be executed by the runtime system
        ///         during system startup.
        virtual hpx::program_options::options_description
            add_commandline_options() = 0;
    };
}}

///////////////////////////////////////////////////////////////////////////////
/// The macro \a HPX_REGISTER_COMMANDLINE_REGISTRY is used to register the given
/// component factory with Hpx.Plugin. This macro has to be used for each of
/// the components.
#define HPX_REGISTER_COMMANDLINE_REGISTRY(RegistryType, componentname)        \
    HPX_PLUGIN_EXPORT(HPX_PLUGIN_COMPONENT_PREFIX,                            \
        hpx::components::component_commandline_base, RegistryType,            \
        componentname, commandline_options)                                   \
/**/
#define HPX_REGISTER_COMMANDLINE_REGISTRY_DYNAMIC(RegistryType, componentname)\
    HPX_PLUGIN_EXPORT_DYNAMIC(HPX_PLUGIN_COMPONENT_PREFIX,                    \
        hpx::components::component_commandline_base, RegistryType,            \
        componentname, commandline_options)                                   \
/**/

/// The macro \a HPX_REGISTER_COMMANDLINE_OPTIONS is used to define the
/// required Hpx.Plugin entry point for the command line option registry.
/// This macro has to be used in not more than one compilation unit of a
/// component module.
#define HPX_REGISTER_COMMANDLINE_OPTIONS()                                    \
    HPX_PLUGIN_EXPORT_LIST(HPX_PLUGIN_COMPONENT_PREFIX, commandline_options); \
    HPX_INIT_REGISTRY_COMMANDLINE_STATIC(HPX_PLUGIN_COMPONENT_PREFIX,         \
        commandline_options)                                                  \
/**/
#define HPX_REGISTER_COMMANDLINE_OPTIONS_DYNAMIC()                            \
    HPX_PLUGIN_EXPORT_LIST_DYNAMIC(HPX_PLUGIN_COMPONENT_PREFIX,               \
        commandline_options)                                                  \
/**/


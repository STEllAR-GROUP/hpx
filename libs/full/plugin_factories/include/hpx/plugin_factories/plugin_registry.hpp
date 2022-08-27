//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/ini.hpp>
#include <hpx/modules/prefix.hpp>
#include <hpx/modules/preprocessor.hpp>
#include <hpx/modules/runtime_configuration.hpp>
#include <hpx/modules/string_util.hpp>
#include <hpx/plugin/traits/plugin_config_data.hpp>

#include <hpx/plugin_factories/unique_plugin_name.hpp>

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::plugins {

    ///////////////////////////////////////////////////////////////////////////
    /// The \a plugin_registry provides a minimal implementation of a
    /// plugin's registry. If no additional functionality is required this
    /// type can be used to implement the full set of minimally required
    /// functions to be exposed by a plugin's registry instance.
    ///
    /// \tparam Plugin   The plugin type this registry should be responsible for.
    template <typename Plugin, char const* const Name,
        char const* const Section, char const* const Suffix>
    struct plugin_registry : public plugin_registry_base
    {
        /// \brief Return the ini-information for all contained components
        ///
        /// \param fillini  [in] The module is expected to fill this vector
        ///                 with the ini-information (one line per vector
        ///                 element) for all components implemented in this
        ///                 module.
        ///
        /// \return Returns \a true if the parameter \a fillini has been
        ///         successfully initialized with the registry data of all
        ///         implemented in this module.
        bool get_plugin_info(std::vector<std::string>& fillini) override
        {
            fillini.emplace_back(std::string("[") + Section + ".plugins." +
                unique_plugin_name<plugin_registry>::call() + "]");
            fillini.emplace_back(std::string("name = ") + Name);
            fillini.emplace_back(std::string("path = ") +
                util::find_prefixes(std::string("/") + Suffix, Name));
            fillini.emplace_back("enabled = 1");

            char const* more = traits::plugin_config_data<Plugin>::call();
            if (more != nullptr)    // -V547
            {
                std::vector<std::string> data;
                hpx::string_util::split(
                    data, more, hpx::string_util::is_any_of("\n"));
                std::copy(
                    data.begin(), data.end(), std::back_inserter(fillini));
            }
            return true;
        }
    };
}    // namespace hpx::plugins

///////////////////////////////////////////////////////////////////////////////
/// This macro is used create and to register a minimal plugin registry with
/// Hpx.Plugin.
#define HPX_REGISTER_PLUGIN_REGISTRY(...)                                      \
    HPX_REGISTER_PLUGIN_REGISTRY_(__VA_ARGS__)                                 \
    /**/

#define HPX_REGISTER_PLUGIN_REGISTRY_(...)                                     \
    HPX_PP_EXPAND(HPX_PP_CAT(HPX_REGISTER_PLUGIN_REGISTRY_,                    \
        HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))                               \
    /**/

#define HPX_REGISTER_PLUGIN_REGISTRY_2(PluginType, pluginname)                 \
    HPX_REGISTER_PLUGIN_REGISTRY_5(                                            \
        PluginType, pluginname, HPX_PLUGIN_NAME, "hpx", "hpx")                 \
    /**/
#define HPX_REGISTER_PLUGIN_REGISTRY_4(                                        \
    PluginType, pluginname, pluginsection, pluginsuffix)                       \
    HPX_REGISTER_PLUGIN_REGISTRY_5(                                            \
        PluginType, pluginname, HPX_PLUGIN_NAME, pluginsection, pluginsuffix)  \
    /**/
#define HPX_REGISTER_PLUGIN_REGISTRY_5(                                        \
    PluginType, pluginname, pluginstring, pluginsection, pluginsuffix)         \
    constexpr char __##pluginname##_string[] = HPX_PP_STRINGIZE(pluginstring); \
    constexpr char __##pluginname##_section[] = pluginsection;                 \
    constexpr char __##pluginname##_suffix[] = pluginsuffix;                   \
    using __##pluginname##_plugin_registry_type =                              \
        hpx::plugins::plugin_registry<PluginType, __##pluginname##_string,     \
            __##pluginname##_section, __##pluginname##_suffix>;                \
    HPX_REGISTER_PLUGIN_BASE_REGISTRY(                                         \
        __##pluginname##_plugin_registry_type, pluginname)                     \
    HPX_DEF_UNIQUE_PLUGIN_NAME(                                                \
        __##pluginname##_plugin_registry_type, pluginname)                     \
    template struct hpx::plugins::plugin_registry<PluginType,                  \
        __##pluginname##_string, __##pluginname##_section,                     \
        __##pluginname##_suffix>;                                              \
    /**/

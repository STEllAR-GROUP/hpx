//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// make inspect happy: hpxinspect:nodeprecatedname:boost::is_any_of

#if !defined(HPX_PLUGIN_REGISTRY_MAR_24_2013_0235PM)
#define HPX_PLUGIN_REGISTRY_MAR_24_2013_0235PM

#include <hpx/config.hpp>
#include <hpx/plugins/plugin_registry_base.hpp>
#include <hpx/plugins/unique_plugin_name.hpp>

#include <hpx/util/detail/pp/cat.hpp>
#include <hpx/util/detail/pp/expand.hpp>
#include <hpx/util/detail/pp/nargs.hpp>
#include <hpx/util/detail/pp/stringize.hpp>
#include <hpx/util/find_prefix.hpp>
#include <hpx/util/ini.hpp>

#include <hpx/traits/plugin_config_data.hpp>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/assign/std/vector.hpp>

#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a plugin_registry provides a minimal implementation of a
    /// plugin's registry. If no additional functionality is required this
    /// type can be used to implement the full set of minimally required
    /// functions to be exposed by a plugin's registry instance.
    ///
    /// \tparam Plugin   The plugin type this registry should be responsible for.
    template <typename Plugin>
    struct plugin_registry : public plugin_registry_base
    {
        ///
        ~plugin_registry() {}

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
        bool get_plugin_info(std::vector<std::string>& fillini)
        {
            using namespace boost::assign;
            fillini += std::string("[hpx.plugins.") +
                unique_plugin_name<plugin_registry>::call() + "]";
            fillini += "name = " HPX_PLUGIN_STRING;
            fillini += std::string("path = ") +
                util::find_prefixes("/hpx", HPX_PLUGIN_STRING);
            fillini += "enabled = 1";

            char const* more = traits::plugin_config_data<Plugin>::call();
            if (more) {
                std::vector<std::string> data;
                boost::split(data, more, boost::is_any_of("\n"));
                std::copy(data.begin(), data.end(), std::back_inserter(fillini));
            }
            return true;
        }
    };
}}

///////////////////////////////////////////////////////////////////////////////
/// This macro is used create and to register a minimal plugin registry with
/// Hpx.Plugin.

#define HPX_REGISTER_PLUGIN_REGISTRY(...)                                     \
        HPX_REGISTER_PLUGIN_REGISTRY_(__VA_ARGS__)                            \
    /**/

#define HPX_REGISTER_PLUGIN_REGISTRY_(...)                                    \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                 \
        HPX_REGISTER_PLUGIN_REGISTRY_, HPX_PP_NARGS(__VA_ARGS__)              \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_PLUGIN_REGISTRY_2(PluginType, pluginname)                \
    typedef hpx::plugins::plugin_registry<PluginType>                         \
        pluginname ## _plugin_registry_type;                                  \
    HPX_REGISTER_PLUGIN_BASE_REGISTRY(                                        \
        pluginname ## _plugin_registry_type, pluginname)                      \
    HPX_DEF_UNIQUE_PLUGIN_NAME(                                               \
        pluginname ## _plugin_registry_type, pluginname)                      \
    template struct hpx::plugins::plugin_registry<PluginType >;               \
/**/

#endif


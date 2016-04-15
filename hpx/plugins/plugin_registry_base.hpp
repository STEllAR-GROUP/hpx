//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PLUGIN_REGISTRY_BASE_MAR_24_2013_0225PM)
#define HPX_PLUGIN_REGISTRY_BASE_MAR_24_2013_0225PM

#include <hpx/config.hpp>

#include <hpx/util/plugin.hpp>
#include <hpx/util/plugin/export_plugin.hpp>
#include <boost/mpl/list.hpp>

#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    struct command_line_handling;
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a plugin_registry_base has to be used as a base class for all
    /// plugin registries.
    struct HPX_EXPORT plugin_registry_base
    {
        virtual ~plugin_registry_base() {}

        /// Return the configuration information for any plugin implemented by
        /// this module
        ///
        /// \param fillini  [in, out] The module is expected to fill this vector
        ///                 with the ini-information (one line per vector
        ///                 element) for all plugins implemented in this
        ///                 module.
        ///
        /// \return Returns \a true if the parameter \a fillini has been
        ///         successfully initialized with the registry data of all
        ///         implemented in this module.
        virtual bool get_plugin_info(std::vector<std::string>& fillini) = 0;

        virtual void init(int *argc, char ***argv, util::command_line_handling&)
        {
        }
    };
}}

///////////////////////////////////////////////////////////////////////////////
/// This macro is used to register the given component factory with
/// Hpx.Plugin. This macro has to be used for each of the components.
#define HPX_REGISTER_PLUGIN_BASE_REGISTRY(PluginType, name)                   \
    HPX_PLUGIN_EXPORT(HPX_PLUGIN_PLUGIN_PREFIX,                               \
        hpx::plugins::plugin_registry_base, PluginType, name, plugin)         \
/**/

/// This macro is used to define the required Hpx.Plugin entry points. This
/// macro has to be used in exactly one compilation unit of a component module.
#define HPX_REGISTER_PLUGIN_REGISTRY_MODULE()                                 \
    HPX_PLUGIN_EXPORT_LIST(HPX_PLUGIN_PLUGIN_PREFIX, plugin)                  \
/**/

#define HPX_REGISTER_PLUGIN_REGISTRY_MODULE_DYNAMIC()                         \
    HPX_PLUGIN_EXPORT_LIST_DYNAMIC(HPX_PLUGIN_PLUGIN_PREFIX, plugin)          \
/**/

#endif


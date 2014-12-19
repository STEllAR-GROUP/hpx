//  Copyright (c)      2014 Thomas Heller
//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PLUGINS_PARCELPORT_FACTORY_BASE_HPP)
#define HPX_PLUGINS_PARCELPORT_FACTORY_BASE_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/plugins/plugin_factory_base.hpp>

#include <hpx/util/plugin.hpp>
#include <hpx/util/plugin/export_plugin.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a plugin_factory_base has to be used as a base class for all
    /// plugin factories.
    struct HPX_EXPORT parcelport_factory_base : plugin_factory_base
    {
        virtual ~parcelport_factory_base() {}

        /// Create a new instance of a parcelport
        ///
        /// return Returns the newly created instance of the parcelport
        ///        supported by this factory
        virtual parcelset::parcelport* create(
            hpx::util::runtime_configuration const & cfg,
            HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
            HPX_STD_FUNCTION<void()> const& on_stop_thread) = 0;
    };
}}

///////////////////////////////////////////////////////////////////////////////
/// This macro is used to register the given component factory with
/// Hpx.Plugin. This macro has to be used for each of the component factories.
#define HPX_REGISTER_PARCELPORT_BASE(FactoryType, pluginname)                 \
    HPX_PLUGIN_EXPORT(HPX_PLUGIN_PLUGIN_PREFIX,                               \
        hpx::plugins::plugin_factory_base, FactoryType,                       \
        pluginname, factory)                                                  \
/**/

#endif

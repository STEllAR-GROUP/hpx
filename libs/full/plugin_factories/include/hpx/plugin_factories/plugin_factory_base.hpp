//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/ini.hpp>
#include <hpx/modules/plugin.hpp>
#include <hpx/modules/type_support.hpp>

#include <hpx/runtime_configuration/plugin_registry_base.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::plugins {

    ///////////////////////////////////////////////////////////////////////////
    /// The \a plugin_factory_base has to be used as a base class for all
    /// plugin factories.
    struct HPX_EXPORT plugin_factory_base
    {
        virtual ~plugin_factory_base() = default;
    };
}    // namespace hpx::plugins

namespace hpx::util::plugin {

    ///////////////////////////////////////////////////////////////////////////
    // The following specialization of the virtual_constructor template
    // defines the argument list for the constructor of the concrete component
    // factory (derived from the component_factory_base above). This magic is needed
    // because we use hpx::plugin for the creation of instances of derived
    // types using the component_factory_base virtual base class only (essentially
    // implementing a virtual constructor).
    //
    // All derived component factories have to expose a constructor with the
    // matching signature. For instance:
    //
    //     class my_factory : public plugin_factory_base
    //     {
    //     public:
    //         my_factory (hpx::util::section const*,
    //              hpx::util::section const*, bool)
    //         {}
    //     };
    //
    template <>
    struct virtual_constructor<hpx::plugins::plugin_factory_base>
    {
        using type = hpx::util::pack<hpx::util::section const*,
            hpx::util::section const*, bool>;
    };
}    // namespace hpx::util::plugin

///////////////////////////////////////////////////////////////////////////////
/// This macro is used to register the given component factory with
/// Hpx.Plugin. This macro has to be used for each of the component factories.
#define HPX_REGISTER_PLUGIN_FACTORY_BASE(FactoryType, pluginname)              \
    HPX_PLUGIN_EXPORT(HPX_PLUGIN_PLUGIN_PREFIX,                                \
        hpx::plugins::plugin_factory_base, FactoryType, pluginname, factory)   \
/**/

/// This macro is used to define the required Hpx.Plugin entry points. This
/// macro has to be used in exactly one compilation unit of a component module.
#define HPX_REGISTER_PLUGIN_MODULE()                                           \
    HPX_PLUGIN_EXPORT_LIST(HPX_PLUGIN_PLUGIN_PREFIX, factory)                  \
    HPX_REGISTER_PLUGIN_REGISTRY_MODULE()                                      \
    /**/

#define HPX_REGISTER_PLUGIN_MODULE_DYNAMIC()                                   \
    HPX_PLUGIN_EXPORT_LIST(HPX_PLUGIN_PLUGIN_PREFIX, factory)                  \
    HPX_REGISTER_PLUGIN_REGISTRY_MODULE_DYNAMIC()                              \
    /**/

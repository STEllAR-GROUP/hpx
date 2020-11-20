//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/plugin.hpp>
#include <hpx/plugins/plugin_factory_base.hpp>
#include <hpx/serialization/binary_filter.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a plugin_factory_base has to be used as a base class for all
    /// plugin factories.
    struct HPX_EXPORT binary_filter_factory_base : plugin_factory_base
    {
        virtual ~binary_filter_factory_base() {}

        /// Create a new instance of a binary filter
        ///
        /// return Returns the newly created instance of the binary filter
        ///        supported by this factory
        virtual serialization::binary_filter* create(bool compress,
            serialization::binary_filter* next_filter = nullptr) = 0;
    };
}}

///////////////////////////////////////////////////////////////////////////////
/// This macro is used to register the given component factory with
/// Hpx.Plugin. This macro has to be used for each of the component factories.
#define HPX_REGISTER_BINARY_FILTER_FACTORY_BASE(FactoryType, pluginname)      \
    HPX_PLUGIN_EXPORT(HPX_PLUGIN_PLUGIN_PREFIX,                               \
        hpx::plugins::plugin_factory_base, FactoryType,                       \
        pluginname, factory)                                                  \
/**/



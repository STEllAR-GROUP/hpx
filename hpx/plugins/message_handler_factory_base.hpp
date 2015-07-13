//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_MESSAGE_HANDLER_FACTORY_BASE_MAR_24_2013_0339PM)
#define HPX_MESSAGE_HANDLER_FACTORY_BASE_MAR_24_2013_0339PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/plugins/plugin_factory_base.hpp>

#include <hpx/util/plugin.hpp>
#include <hpx/util/plugin/export_plugin.hpp>
#include <hpx/runtime/parcelset/policies/message_handler.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a plugin_factory_base has to be used as a base class for all
    /// plugin factories.
    struct HPX_EXPORT message_handler_factory_base : plugin_factory_base
    {
        virtual ~message_handler_factory_base() {}

        /// Create a new instance of a message handler
        ///
        /// return Returns the newly created instance of the message handler
        ///        supported by this factory
        virtual parcelset::policies::message_handler* create(
            char const* action, parcelset::parcelport* pp,
            std::size_t num_messages, std::size_t interval) = 0;
    };
}}

///////////////////////////////////////////////////////////////////////////////
/// This macro is used to register the given component factory with
/// Hpx.Plugin. This macro has to be used for each of the component factories.
#define HPX_REGISTER_MESSAGE_HANDLER_FACTORY_BASE(FactoryType, pluginname)    \
    HPX_PLUGIN_EXPORT(HPX_PLUGIN_PLUGIN_PREFIX,                               \
        hpx::plugins::plugin_factory_base, FactoryType,                       \
        pluginname, factory)                                                  \
/**/

#endif


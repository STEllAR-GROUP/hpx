//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_MESSAGE_HANDLER_FACTORY_MAR_24_2013_0347PM)
#define HPX_MESSAGE_HANDLER_FACTORY_MAR_24_2013_0347PM

#include <hpx/config.hpp>
#include <hpx/plugins/message_handler_factory_base.hpp>
#include <hpx/plugins/plugin_registry.hpp>
#include <hpx/plugins/unique_plugin_name.hpp>

#include <hpx/util/detail/count_num_args.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a message_handler_factory provides a minimal implementation of a
    /// message handler's factory. If no additional functionality is required
    /// this type can be used to implement the full set of minimally required
    /// functions to be exposed by a message handler's factory instance.
    ///
    /// \tparam MessageHandler The message handler type this factory should be
    ///                        responsible for.
    template <typename MessageHandler>
    struct message_handler_factory : public message_handler_factory_base
    {
        /// \brief Construct a new factory instance
        ///
        /// \param global   [in] The pointer to a \a hpx#util#section instance
        ///                 referencing the settings read from the [settings]
        ///                 section of the global configuration file (hpx.ini)
        ///                 This pointer may be nullptr if no such section has
        ///                 been found.
        /// \param local    [in] The pointer to a \a hpx#util#section instance
        ///                 referencing the settings read from the section
        ///                 describing this component type:
        ///                 [hpx.components.\<name\>], where \<name\> is the
        ///                 instance name of the component as given in the
        ///                 configuration files.
        ///
        /// \note The contents of both sections has to be cloned in order to
        ///       save the configuration setting for later use.
        message_handler_factory(util::section const* global,
                util::section const* local, bool isenabled)
          : isenabled_(isenabled)
        {
            // store the configuration settings
            if (nullptr != global)
                global_settings_ = *global;
            if (nullptr != local)
                local_settings_ = *local;
        }

        ///
        ~message_handler_factory() {}

        /// Register a action for this message handler type
        void register_action(char const* action, error_code& ec)
        {
            MessageHandler::register_action(action, ec);
        }

        /// Create a new instance of a message handler
        ///
        /// return Returns the newly created instance of the message handler
        ///        supported by this factory
        parcelset::policies::message_handler* create(char const* action,
            parcelset::parcelport* pp, std::size_t num_messages,
            std::size_t interval)
        {
            if (isenabled_)
                return new MessageHandler(action, pp, num_messages, interval);
            return 0;
        }

    protected:
        util::section global_settings_;
        util::section local_settings_;
        bool isenabled_;
    };
}}

///////////////////////////////////////////////////////////////////////////////
/// This macro is used create and to register a minimal component factory with
/// Hpx.Plugin.
#define HPX_REGISTER_MESSAGE_HANDLER_FACTORY(MessageHandler, pluginname)      \
    HPX_REGISTER_MESSAGE_HANDLER_FACTORY_BASE(                                \
        hpx::plugins::message_handler_factory<MessageHandler>, pluginname)    \
    HPX_DEF_UNIQUE_PLUGIN_NAME(                                               \
        hpx::plugins::message_handler_factory<MessageHandler>, pluginname)    \
    template struct hpx::plugins::message_handler_factory<MessageHandler>;    \
    HPX_REGISTER_PLUGIN_REGISTRY_2(MessageHandler, pluginname)                \
/**/

#endif


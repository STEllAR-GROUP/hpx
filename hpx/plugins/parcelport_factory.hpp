//  Copyright (c)      2014 Thomas Heller
//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2020 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/plugin/traits/plugin_config_data.hpp>
#include <hpx/plugins/parcelport_factory_base.hpp>
#include <hpx/plugins/plugin_factory_base.hpp>
#include <hpx/plugins/unique_plugin_name.hpp>
#include <hpx/prefix/find_prefix.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime_configuration/runtime_configuration.hpp>
#include <hpx/string_util/classification.hpp>
#include <hpx/string_util/split.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins {
    ///////////////////////////////////////////////////////////////////////////
    /// The \a parcelport_factory provides a minimal implementation of a
    /// parcelport factory. If no additional functionality is required
    /// this type can be used to implement the full set of minimally required
    /// functions to be exposed by a parcelport factory instance.
    ///
    /// \tparam Parcelport The parcelport type this factory should be
    ///                        responsible for.
    template <typename Parcelport>
    struct parcelport_factory : public parcelport_factory_base
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
        parcelport_factory()
        {
            parcelset::parcelhandler::add_parcelport_factory(this);
        }

        explicit parcelport_factory(
            std::vector<plugins::parcelport_factory_base*>& factories)
        {
            factories.push_back(this);
        }

        ~parcelport_factory() override
        {
            traits::plugin_config_data<Parcelport>::destroy();
        }

        void get_plugin_info(std::vector<std::string>& fillini) override
        {
            std::string name = unique_plugin_name<parcelport_factory>::call();
            fillini.emplace_back(std::string("[hpx.parcel.") + name + "]");
            fillini.emplace_back("name = " HPX_PLUGIN_STRING);
            fillini.emplace_back(std::string("path = ") +
                util::find_prefixes("/hpx", HPX_PLUGIN_STRING));
            fillini.emplace_back("enable = $[hpx.parcel.enable]");

            std::string name_uc;
            name_uc.reserve(name.size());
            std::transform(name.begin(), name.end(),
                std::back_inserter(name_uc),
                [](char c) { return std::toupper(c); });

            // basic parcelport configuration ...
            fillini.emplace_back("parcel_pool_size = ${HPX_PARCEL_" + name_uc +
                "_PARCEL_POOL_SIZE:"
                "$[hpx.threadpools.parcel_pool_size]}");
            fillini.emplace_back("max_connections =  ${HPX_PARCEL_" + name_uc +
                "_MAX_CONNECTIONS:"
                "$[hpx.parcel.max_connections]}");
            fillini.emplace_back(
                "max_connections_per_locality = ${HPX_PARCEL_" + name_uc +
                "_MAX_CONNECTIONS_PER_LOCALITY:"
                "$[hpx.parcel.max_connections_per_locality]}");
            fillini.emplace_back("max_message_size =  ${HPX_PARCEL_" + name_uc +
                "_MAX_MESSAGE_SIZE:$[hpx.parcel.max_message_size]}");
            fillini.emplace_back("max_outbound_message_size =  ${HPX_PARCEL_" +
                name_uc + "_MAX_OUTBOUND_MESSAGE_SIZE" +
                ":$[hpx.parcel.max_outbound_message_size]}");
            fillini.emplace_back("array_optimization = ${HPX_PARCEL_" +
                name_uc +
                "_ARRAY_OPTIMIZATION:$[hpx.parcel.array_optimization]}");
            fillini.emplace_back("zero_copy_optimization = ${HPX_PARCEL_" +
                name_uc +
                "_ZERO_COPY_OPTIMIZATION:"
                "$[hpx.parcel.zero_copy_optimization]}");
            fillini.emplace_back("async_serialization = ${HPX_PARCEL_" +
                name_uc +
                "_ASYNC_SERIALIZATION:"
                "$[hpx.parcel.async_serialization]}");
            fillini.emplace_back("priority = ${HPX_PARCEL_" + name_uc +
                "_PRIORITY:" +
                traits::plugin_config_data<Parcelport>::priority() + "}");

            // get the parcelport specific information ...
            char const* more = traits::plugin_config_data<Parcelport>::call();
            if (more != nullptr)    // -V547
            {
                std::vector<std::string> data;
                hpx::string_util::split(
                    data, more, hpx::string_util::is_any_of("\n"));
                std::copy(
                    data.begin(), data.end(), std::back_inserter(fillini));
            }
        }

        void init(
            int* argc, char*** argv, util::command_line_handling& cfg) override
        {
            // initialize the parcelport with the parameters we got passed in at start
            traits::plugin_config_data<Parcelport>::init(argc, argv, cfg);
        }

        /// Create a new instance of a message handler
        ///
        /// return Returns the newly created instance of the message handler
        ///        supported by this factory
        parcelset::parcelport* create(
            hpx::util::runtime_configuration const& cfg,
            threads::policies::callback_notifier const& notifier) override
        {
            return new Parcelport(cfg, notifier);
        }
    };
}}    // namespace hpx::plugins

///////////////////////////////////////////////////////////////////////////////
/// This macro is used create and to register a minimal component factory with
/// Hpx.Plugin.
#define HPX_REGISTER_PARCELPORT_(Parcelport, pluginname, pp)                   \
    typedef hpx::plugins::parcelport_factory<Parcelport> HPX_PP_CAT(           \
        pluginname, _plugin_factory_type);                                     \
    HPX_DEF_UNIQUE_PLUGIN_NAME(                                                \
        HPX_PP_CAT(pluginname, _plugin_factory_type), pp)                      \
    template struct hpx::plugins::parcelport_factory<Parcelport>;              \
    HPX_EXPORT hpx::plugins::parcelport_factory_base* HPX_PP_CAT(              \
        pluginname, _factory_init)(                                            \
        std::vector<hpx::plugins::parcelport_factory_base*> & factories)       \
    {                                                                          \
        static HPX_PP_CAT(pluginname, _plugin_factory_type)                    \
            factory(factories);                                                \
        return &factory;                                                       \
    }                                                                          \
    /**/

#define HPX_REGISTER_PARCELPORT(Parcelport, pluginname)                        \
    HPX_REGISTER_PARCELPORT_(                                                  \
        Parcelport, HPX_PP_CAT(parcelport_, pluginname), pluginname)

//  Copyright (c)      2014 Thomas Heller
//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PLUGINS_PARCELPORT_FACTORY_HPP)
#define HPX_PLUGINS_PARCELPORT_FACTORY_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/plugins/unique_plugin_name.hpp>
#include <hpx/plugins/plugin_registry.hpp>
#include <hpx/plugins/parcelport_factory_base.hpp>

#include <hpx/util/plugin.hpp>
#include <hpx/util/plugin/export_plugin.hpp>

#include <boost/preprocessor/cat.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins
{
    template <typename Parcelport>
    struct parcelport_registry : plugin_registry_base
    {
        ///
        ~parcelport_registry() {}

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
            std::string name = unique_plugin_name<parcelport_registry>::call();
            fillini += std::string("[hpx.parcel.") + name + "]";
            fillini += "name = " HPX_PLUGIN_STRING;
            fillini += std::string("path = ") +
                util::find_prefixes("/lib/hpx", HPX_PLUGIN_STRING);
            fillini += "enable = 1";

            std::string name_uc = boost::to_upper_copy(name);
            // basic parcelport configuration ...
            fillini +=
                "parcel_pool_size = ${HPX_PARCEL_" + name_uc + "_PARCEL_POOL_SIZE:"
                    "$[hpx.threadpools.parcel_pool_size]}",
                "max_connections =  ${HPX_PARCEL_" + name_uc + "_MAX_CONNECTIONS:"
                    "$[hpx.parcel.max_connections]}",
                "max_connections_per_locality = "
                    "${HPX_PARCEL_" + name_uc + "_MAX_CONNECTIONS_PER_LOCALITY:"
                    "$[hpx.parcel.max_connections_per_locality]}",
                "max_message_size =  ${HPX_PARCEL_" + name_uc +
                    "_MAX_MESSAGE_SIZE:$[hpx.parcel.max_message_size]}",
                "max_outbound_message_size =  ${HPX_PARCEL_" + name_uc +
                    "_MAX_OUTBOUND_MESSAGE_SIZE:$[hpx.parcel.max_outbound_message_size]}",
                "array_optimization = ${HPX_PARCEL_" + name_uc +
                    "_ARRAY_OPTIMIZATION:$[hpx.parcel.array_optimization]}",
                "zero_copy_optimization = ${HPX_PARCEL_" + name_uc +
                    "_ZERO_COPY_OPTIMIZATION:"
                    "$[hpx.parcel.zero_copy_optimization]}",
                "enable_security = ${HPX_PARCEL_" + name_uc +
                    "_ENABLE_SECURITY:"
                    "$[hpx.parcel.enable_security]}",
                "async_serialization = ${HPX_PARCEL_" + name_uc +
                    "_ASYNC_SERIALIZATION:"
                    "$[hpx.parcel.async_serialization]}",
                "priority = ${HPX_PARCEL_" + name_uc +
                    "_PRIORITY:" + traits::plugin_config_data<Parcelport>::priority() + "}"
                ;

            // get the parcelport specific information ...
            char const* more = traits::plugin_config_data<Parcelport>::call();
            if (more) {
                std::vector<std::string> data;
                boost::split(data, more, boost::is_any_of("\n"));
                std::copy(data.begin(), data.end(), std::back_inserter(fillini));
            }
            return true;
        }

        virtual void init(int *argc, char ***argv, util::command_line_handling &cfg)
        {
            // initialize the parcelport with the parameters we got passed in at start
            traits::plugin_config_data<Parcelport>::init(argc, argv, cfg);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    /// The \a parcelport_factory provides a minimal implementation of a
    /// parcelport's factory. If no additional functionality is required
    /// this type can be used to implement the full set of minimally required
    /// functions to be exposed by a parcelports's factory instance.
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
        ///                 This pointer may be NULL if no such section has
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
        parcelport_factory(util::section const* global,
                util::section const* local, bool isenabled)
          : isenabled_(isenabled)
        {
            // store the configuration settings
            if (NULL != global)
                global_settings_ = *global;
            if (NULL != local)
                local_settings_ = *local;
        }

        ///
        ~parcelport_factory() {}

        /// Create a new instance of a message handler
        ///
        /// return Returns the newly created instance of the message handler
        ///        supported by this factory
        parcelset::parcelport* create(
            hpx::util::runtime_configuration const & cfg,
            HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
            HPX_STD_FUNCTION<void()> const& on_stop_thread)
        {
            if (isenabled_)
                return new Parcelport(cfg, on_start_thread, on_stop_thread);
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
#define HPX_REGISTER_PARCELPORT_(Parcelport, pluginname, pp)                  \
    HPX_REGISTER_PARCELPORT_BASE(                                             \
        hpx::plugins::parcelport_factory<Parcelport>, pluginname)             \
    HPX_DEF_UNIQUE_PLUGIN_NAME(                                               \
        hpx::plugins::parcelport_factory<Parcelport>, pluginname)             \
    template struct hpx::plugins::parcelport_factory<Parcelport>;             \
    typedef hpx::plugins::parcelport_registry<Parcelport>                     \
        BOOST_PP_CAT(pluginname, _plugin_registry_type);                      \
    HPX_REGISTER_PLUGIN_BASE_REGISTRY(                                        \
        BOOST_PP_CAT(pluginname, _plugin_registry_type), pluginname)          \
    HPX_DEF_UNIQUE_PLUGIN_NAME(                                               \
        BOOST_PP_CAT(pluginname, _plugin_registry_type), pp)                  \
    template struct hpx::plugins::parcelport_registry<Parcelport >;           \

#define HPX_REGISTER_PARCELPORT(Parcelport, pluginname)                       \
        HPX_REGISTER_PARCELPORT_(Parcelport, BOOST_PP_CAT(parcelport_, pluginname), pluginname)\

#endif


//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_BINARY_FILTER_FACTORY_MAR_24_2013_0347PM)
#define HPX_BINARY_FILTER_FACTORY_MAR_24_2013_0347PM

#include <hpx/config.hpp>
#include <hpx/plugins/unique_plugin_name.hpp>
#include <hpx/plugins/plugin_registry.hpp>
#include <hpx/plugins/binary_filter_factory_base.hpp>

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
    /// \tparam BinaryFilter The message handler type this factory should be
    ///                        responsible for.
    template <typename BinaryFilter>
    struct binary_filter_factory : public binary_filter_factory_base
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
        binary_filter_factory(util::section const* global,
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
        ~binary_filter_factory() {}

        /// Create a new instance of a message handler
        ///
        /// return Returns the newly created instance of the message handler
        ///        supported by this factory
        serialization::binary_filter* create(bool compress,
            serialization::binary_filter* next_filter = 0)
        {
            if (isenabled_)
                return new BinaryFilter(compress, next_filter);
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
#define HPX_REGISTER_BINARY_FILTER_FACTORY(BinaryFilter, pluginname)          \
    HPX_REGISTER_BINARY_FILTER_FACTORY_BASE(                                  \
        hpx::plugins::binary_filter_factory<BinaryFilter>, pluginname)        \
    HPX_DEF_UNIQUE_PLUGIN_NAME(                                               \
        hpx::plugins::binary_filter_factory<BinaryFilter>, pluginname)        \
    template struct hpx::plugins::binary_filter_factory<BinaryFilter>;        \
    HPX_REGISTER_PLUGIN_REGISTRY_2(BinaryFilter, pluginname)                  \
/**/

#endif


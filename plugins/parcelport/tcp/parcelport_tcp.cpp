//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <hpx/plugins/parcelport/tcp/connection_handler.hpp>
#include <hpx/plugins/parcelport/tcp/sender.hpp>
#include <hpx/plugins/parcelport_factory.hpp>

namespace hpx { namespace traits
{
    // Inject additional configuration data into the factory registry for this
    // type. This information ends up in the system wide configuration database
    // under the plugin specific section:
    //
    //      [hpx.parcel.tcp]
    //      ...
    //      priority = 1
    //
    template <>
    struct plugin_config_data<hpx::parcelset::policies::tcp::connection_handler>
    {
        static char const* priority()
        {
            return "1";
        }

        static void init(int *argc, char ***argv, util::command_line_handling &cfg)
        {
        }
        static char const* call()
        {
            return "";
        }
    };
}}

HPX_REGISTER_PARCELPORT(
    hpx::parcelset::policies::tcp::connection_handler,
    tcp);

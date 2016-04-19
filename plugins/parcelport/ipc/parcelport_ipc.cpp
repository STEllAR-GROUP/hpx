//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <hpx/plugins/parcelport/ipc/connection_handler.hpp>

#include <hpx/plugins/parcelport_factory.hpp>

namespace hpx { namespace traits
{
    // Inject additional configuration data into the factory registry for this
    // type. This information ends up in the system wide configuration database
    // under the plugin specific section:
    //
    //      [hpx.parcel.ipc]
    //      ...
    //      priority = 30
    //
    template <>
    struct plugin_config_data<hpx::parcelset::policies::ipc::connection_handler>
    {
        static char const* priority()
        {
            return "30";
        }

        static void init(int *argc, char ***argv, util::command_line_handling &cfg)
        {
        }
        static char const* call()
        {
            return
                "data_buffer_cache_size = ${HPX_PARCEL_IPC_DATA_BUFFER_CACHE_SIZE:512}\n"
                "zero_copy_optimization = 0\n"
                "async_serialization = 0\n"
                "enable = 0"
                ;
        }
    };
}}

HPX_REGISTER_PLUGIN_MODULE();
HPX_REGISTER_PARCELPORT(
    hpx::parcelset::policies::ipc::connection_handler,
    ipc);

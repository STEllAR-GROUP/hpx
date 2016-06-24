//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/traits/plugin_config_data.hpp>

#include <hpx/plugins/parcelport/ibverbs/connection_handler.hpp>

#include <hpx/plugins/parcelport_factory.hpp>

namespace hpx { namespace traits
{
    // Inject additional configuration data into the factory registry for this
    // type. This information ends up in the system wide configuration database
    // under the plugin specific section:
    //
    //      [hpx.parcel.ibverbs]
    //      ...
    //      priority = 50
    //
    template <>
    struct plugin_config_data<hpx::parcelset::policies::ibverbs::connection_handler>
    {
        static char const* priority()
        {
            return "20";
        }

        static void init(int *argc, char ***argv, util::command_line_handling &cfg)
        {
        }

        static char const* call()
        {
            return
                "ifname = ${HPX_HAVE_PARCEL_IBVERBS_IFNAME:"
                HPX_HAVE_PARCELPORT_IBVERBS_IFNAME "}\n"
                "memory_chunk_size = ${HPX_HAVE_PARCEL_IBVERBS_MEMORY_CHUNK_SIZE:"
                    BOOST_PP_STRINGIZE(HPX_HAVE_PARCELPORT_IBVERBS_MEMORY_CHUNK_SIZE)
                "}\n"
                "max_memory_chunks = ${HPX_HAVE_PARCEL_IBVERBS_MAX_MEMORY_CHUNKS:"
                    BOOST_PP_STRINGIZE(HPX_HAVE_PARCELPORT_IBVERBS_MAX_MEMORY_CHUNKS)
                "}\n"
                "zero_copy_optimization = 0\n"
                "io_pool_size = 2\n"
                "use_io_pool = 1\n"
                "enable = 0"
                ;
        }
    };
}}

HPX_REGISTER_PLUGIN_MODULE();
HPX_REGISTER_PARCELPORT(
    hpx::parcelset::policies::ibverbs::connection_handler,
    ibverbs);

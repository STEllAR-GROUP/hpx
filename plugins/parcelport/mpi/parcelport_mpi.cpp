//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#include <hpx/plugins/parcelport/mpi/connection_handler.hpp>
#include <hpx/plugins/parcelport/mpi/mpi_environment.hpp>

#include <hpx/plugins/parcelport_factory.hpp>

#include <hpx/util/command_line_handling.hpp>

namespace hpx { namespace traits
{
    // Inject additional configuration data into the factory registry for this
    // type. This information ends up in the system wide configuration database
    // under the plugin specific section:
    //
    //      [hpx.parcel.mpi]
    //      ...
    //      priority = 100
    //
    template <>
    struct plugin_config_data<hpx::parcelset::policies::mpi::connection_handler>
    {
        static char const* priority()
        {
            return "10";
        }
        static void init(int *argc, char ***argv, util::command_line_handling &cfg)
        {
            util::mpi_environment::init(argc, argv, cfg);
        }

        static char const* call()
        {
            return
#if defined(HPX_PARCELPORT_MPI_ENV)
                "env = ${HPX_PARCELPORT_MPI_ENV:" HPX_PARCELPORT_MPI_ENV "}\n"
#else
                "env = ${HPX_PARCELPORT_MPI_ENV:PMI_RANK,OMPI_COMM_WORLD_SIZE}\n"
#endif
                "multithreaded = ${HPX_PARCELPORT_MPI_MULTITHREADED:0}\n"
                "io_pool_size = 1\n"
                "use_io_pool = 1\n"
                ;
        }
    };
}}

HPX_REGISTER_PLUGIN_MODULE();
HPX_REGISTER_PARCELPORT(
    hpx::parcelset::policies::mpi::connection_handler,
    mpi);

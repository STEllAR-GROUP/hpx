//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if defined(HPX_HAVE_PARCELPORT_MPI)

#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/command_line_handling.hpp>
#include <hpx/util/mpi_environment.hpp>

#include <boost/format.hpp>

#include <iostream>

namespace hpx { namespace util
{
    MPI_Comm mpi_environment::communicator_ = reinterpret_cast<MPI_Comm>(-1);

    void mpi_environment::init(int *argc, char ***argv, command_line_handling& cfg)
    {
        std::string bootstrap_parcelport = cfg.rtcfg_.get_entry("hpx.parcel.bootstrap", "tcpip");

        bool enable_mpi = false;
        if(bootstrap_parcelport == "mpi")
        {
            cfg.rtcfg_.parse("mpi enable", "hpx.parcel.mpi.enable!=1");
            enable_mpi = true;
        }
        else
        {
            std::string enable_mpi_str = cfg.rtcfg_.get_entry("hpx.parcel.mpi.enable", "0");
            enable_mpi = boost::lexical_cast<int>(enable_mpi_str);
            if(enable_mpi)
            {
                cfg.rtcfg_.parse("mpi enable", "hpx.parcel.bootstrap!=mpi");
            }
        }

        if (enable_mpi)
        {
            MPI_Init(argc, argv);
            MPI_Comm_dup(MPI_COMM_WORLD, &communicator_);
            int this_rank = rank();
            cfg.rtcfg_.parse("mpi rank",
                boost::str(boost::format("hpx.locality!=%1%")
                          % this_rank));
            cfg.rtcfg_.parse("mpi size",
                boost::str(boost::format("hpx.localities!=%1%")
                          % size()));

            if(this_rank == 0)
            {
                cfg.mode_ = hpx::runtime_mode_console;
                cfg.rtcfg_.parse("mpi service mode", "hpx.agas.service_mode!=bootstrap");
            }
            else
            {
                cfg.mode_ = hpx::runtime_mode_worker;
                cfg.rtcfg_.parse("mpi service mode", "hpx.agas.service_mode!=hosted");
            }
            cfg.rtcfg_.parse("mpi runtime mode",
                boost::str(boost::format("hpx.runtime_mode!=%1%")
                          % get_runtime_mode_name(cfg.mode_)));
            std::cout << communicator_ << "\n";
        }
    }

    void mpi_environment::finalize()
    {
        if(enabled())
        {
            std::cout << "Rank " << rank() << " finalizing\n";
            MPI_Comm_free(&communicator_);
            MPI_Finalize();
        }
    }

    MPI_Comm &mpi_environment::communicator()
    {
        return communicator_;
    }

    bool mpi_environment::enabled()
    {
        return communicator_ != reinterpret_cast<MPI_Comm>(-1);
    }

    int mpi_environment::size()
    {
        int res(-1);
        if(!enabled()) return res;
        MPI_Comm_size(communicator_, &res);
        return res;
    }

    int mpi_environment::rank()
    {
        int res(-1);
        if(!enabled()) return res;
        MPI_Comm_rank(communicator_, &res);
        return res;
    }
}}

#endif

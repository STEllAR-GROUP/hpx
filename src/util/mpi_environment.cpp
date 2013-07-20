//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if defined(HPX_HAVE_PARCELPORT_MPI)

#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/command_line_handling.hpp>
#include <hpx/util/mpi_environment.hpp>

#include <boost/format.hpp>
#include <boost/assign.hpp>

#include <iostream>

#include <iostream>

namespace hpx { namespace util
{
    MPI_Comm mpi_environment::communicator_ = 0;

    void mpi_environment::init(int *argc, char ***argv, command_line_handling& cfg)
    {
        using namespace boost::assign;
        std::string bootstrap_parcelport =
            cfg.rtcfg_.get_entry("hpx.parcel.bootstrap", "tcpip");

        bool enable_mpi = false;
        if(bootstrap_parcelport == "mpi")
        {
            cfg.ini_config_ += "hpx.parcel.mpi.enable!=1";
            enable_mpi = true;
        }
        else
        {
            std::string enable_mpi_str =
                cfg.rtcfg_.get_entry("hpx.parcel.mpi.enable", "0");
            enable_mpi = boost::lexical_cast<int>(enable_mpi_str);
            if(enable_mpi)
            {
                cfg.ini_config_ += "hpx.parcel.bootstrap!=mpi";
            }
        }

        if (enable_mpi)
        {
            MPI_Init(argc, argv);
            MPI_Comm_dup(MPI_COMM_WORLD, &communicator_);

            char name[MPI_MAX_PROCESSOR_NAME] = { '\0' };
            int len = 0;
            MPI_Get_processor_name(name, &len);

            std::cout << rank() << ": " << name << "\n";
            std::cout << communicator_ << "\n";

            int this_rank = rank();
            cfg.ini_config_ += "hpx.locality!=" + boost::lexical_cast<std::string>(this_rank);
            cfg.ini_config_ += "hpx.localities!=" + boost::lexical_cast<std::string>(size());
            cfg.num_localities_ = size();

            std::cout << rank() << " mpi_environment::init(): " << size() << "\n";

            if(this_rank == 0)
            {
                cfg.mode_ = hpx::runtime_mode_console;
                cfg.ini_config_ += "hpx.agas.service_mode!=bootstrap";
            }
            else
            {
                cfg.mode_ = hpx::runtime_mode_worker;
                cfg.ini_config_ += "hpx.agas.service_mode!=hosted";
                cfg.hpx_main_f_ = 0;
            }
            cfg.ini_config_ += std::string("hpx.runtime_mode!=") +
                get_runtime_mode_name(cfg.mode_);

            cfg.rtcfg_.reconfigure(cfg.ini_config_);
        }
    }

    void mpi_environment::finalize()
    {
        if(enabled())
        {
            std::cout << "Rank " << rank() << " finalizing\n";

            MPI_Comm communicator = communicator_;
            communicator_ = 0;
            MPI_Comm_free(&communicator);

            MPI_Finalize();
        }
    }

    MPI_Comm &mpi_environment::communicator()
    {
        return communicator_;
    }

    bool mpi_environment::enabled()
    {
        return communicator_ != 0;
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

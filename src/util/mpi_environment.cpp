//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCELPORT_MPI)

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/command_line_handling.hpp>
#include <hpx/util/mpi_environment.hpp>

#include <boost/assign/std/vector.hpp>
#include <boost/tokenizer.hpp>

#include <cstdlib>

namespace hpx { namespace util
{
    namespace detail
    {
        bool detect_mpi_environment(util::runtime_configuration const& cfg)
        {
            std::string mpi_environment_strings = cfg.get_entry(
                "hpx.parcel.mpi.env", HPX_PARCELPORT_MPI_ENV);

            typedef
                boost::tokenizer<boost::char_separator<char> >
                tokenizer;
            boost::char_separator<char> sep(";,: ");
            tokenizer tokens(mpi_environment_strings, sep);
            for(tokenizer::iterator it = tokens.begin(); it != tokens.end(); ++it)
            {
                char *env = std::getenv(it->c_str());
                if(env) return true;
            }
            return false;
        }
    }

    MPI_Comm mpi_environment::communicator_ = 0;

    int mpi_environment::init(int *argc, char ***argv, command_line_handling& cfg)
    {
        using namespace boost::assign;

        bool enable_mpi
            = boost::lexical_cast<int>(
                cfg.rtcfg_.get_entry("hpx.parcel.mpi.enable", "1")
            );

        enable_mpi = enable_mpi && detail::detect_mpi_environment(cfg.rtcfg_);
        if(!enable_mpi) return -1;

        cfg.ini_config_ += "hpx.parcel.bootstrap!=mpi";

        int this_rank = -1;

        MPI_Init(argc, argv);
        MPI_Comm_dup(MPI_COMM_WORLD, &communicator_);

        char name[MPI_MAX_PROCESSOR_NAME+1] = { '\0' };
        int len = 0;
        MPI_Get_processor_name(name, &len);

        this_rank = rank();
        cfg.num_localities_ = static_cast<std::size_t>(size());

        if(this_rank == 0)
        {
            cfg.mode_ = hpx::runtime_mode_console;
        }
        else
        {
            cfg.mode_ = hpx::runtime_mode_worker;
        }

        cfg.ini_config_ += std::string("hpx.hpx.parcel.mpi.rank!=") +
            boost::lexical_cast<std::string>(this_rank);
        cfg.ini_config_ += std::string("hpx.hpx.parcel.mpi.processorname!=") +
            name;

        return this_rank;
    }

    void mpi_environment::finalize()
    {
        if(enabled())
        {
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
        if(enabled())
            MPI_Comm_size(communicator_, &res);
        return res;
    }

    int mpi_environment::rank()
    {
        int res(-1);
        if(enabled())
            MPI_Comm_rank(communicator_, &res);
        return res;
    }
}}

#endif

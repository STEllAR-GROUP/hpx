//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config/defines.hpp>

#if defined(HPX_HAVE_PARCELPORT_MPI)
#include <mpi.h>
#include <hpx/config.hpp>

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
#if defined(__bgq__)
            // If running on BG/Q, we can safely assume to always run in an
            // MPI environment
            return true;
#else
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
#endif
        }

        int get_cfg_entry(command_line_handling& cfg, std::string const& str,
            int dflt)
        {
            try {
                return boost::lexical_cast<int>(
                    cfg.rtcfg_.get_entry(str, dflt));
            }
            catch (boost::bad_lexical_cast const&) {
                /**/;
            }
            return dflt;
        }
    }

    bool mpi_environment::enabled_ = false;

    int mpi_environment::init(int *argc, char ***argv, command_line_handling& cfg)
    {
        using namespace boost::assign;

        int this_rank = -1;

        // We assume to use the MPI parcelport if it is not explicitly disabled
        enabled_ = detail::get_cfg_entry(cfg, "hpx.parcel.mpi.enable", 1);
        if (!enabled_) return this_rank;

        // We disable the MPI parcelport if the application is not run using mpirun
        // and the tcp/ip parcelport is not explicitly disabled
        //
        // The bottomline is that we use the MPI parcelport either when the application
        // was executed using mpirun or if the tcp/ip parcelport was disabled.
        if (!detail::detect_mpi_environment(cfg.rtcfg_) &&
            detail::get_cfg_entry(cfg, "hpx.parcel.tcpip.enable", 1))
        {
            enabled_ = false;
            return this_rank;
        }

        cfg.ini_config_ += "hpx.parcel.bootstrap!=mpi";

        MPI_Init(argc, argv);

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
            get_processor_name();

        return this_rank;
    }

    std::string mpi_environment::get_processor_name()
    {
        char name[MPI_MAX_PROCESSOR_NAME + 1] = { '\0' };
        int len = 0;
        MPI_Get_processor_name(name, &len);

        return name;
    }

    void mpi_environment::finalize()
    {
        if(enabled())
        {
            MPI_Finalize();
        }
    }

    bool mpi_environment::enabled()
    {
        return enabled_;
    }

    int mpi_environment::size()
    {
        int res(-1);
        if(enabled())
            MPI_Comm_size(MPI_COMM_WORLD, &res);
        return res;
    }

    int mpi_environment::rank()
    {
        int res(-1);
        if(enabled())
            MPI_Comm_rank(MPI_COMM_WORLD, &res);
        return res;
    }
}}

#endif

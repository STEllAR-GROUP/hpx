//  Copyright (c) 2013-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <mpi.h>

#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/command_line_handling.hpp>
#include <hpx/plugins/parcelport/mpi/mpi_environment.hpp>

#include <boost/assign/std/vector.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>

#include <cstdlib>
#include <string>

namespace hpx { namespace util
{
    namespace detail
    {
        bool detect_mpi_environment(util::runtime_configuration const& cfg,
            char const* default_env)
        {
#if defined(__bgq__)
            // If running on BG/Q, we can safely assume to always run in an
            // MPI environment
            return true;
#else
            std::string mpi_environment_strings = cfg.get_entry(
                "hpx.parcel.mpi.env", default_env);

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

        int get_cfg_entry(runtime_configuration const& cfg,
            std::string const& str, int dflt)
        {
            try {
                return boost::lexical_cast<int>(
                    cfg.get_entry(str, dflt));
            }
            catch (boost::bad_lexical_cast const&) {
                /**/;
            }
            return dflt;
        }

        int get_cfg_entry(command_line_handling& cfg, std::string const& str,
            int dflt)
        {
            return get_cfg_entry(cfg.rtcfg_, str, dflt);
        }
    }
}}

namespace hpx { namespace util
{
    mpi_environment::mutex_type mpi_environment::mtx_;
    bool mpi_environment::enabled_ = false;
    bool mpi_environment::has_called_init_ = false;
    int mpi_environment::provided_threading_flag_ = MPI_THREAD_SINGLE;
    MPI_Comm mpi_environment::communicator_ = MPI_COMM_NULL;

    ///////////////////////////////////////////////////////////////////////////
    bool mpi_environment::check_mpi_environment(runtime_configuration const& cfg)
    {
        if (detail::get_cfg_entry(cfg, "hpx.parcel.mpi.enable", 1) == 0)
            return false;

        // We disable the MPI parcelport if the application is not run using
        // mpirun and the tcp/ip parcelport is not explicitly disabled
        //
        // The bottom line is that we use the MPI parcelport either when the
        // application was executed using mpirun or if the tcp/ip parcelport
        // was disabled.
        if (!detail::detect_mpi_environment(cfg, HPX_HAVE_PARCELPORT_MPI_ENV) &&
            detail::get_cfg_entry(cfg, "hpx.parcel.tcp.enable", 1))
        {
            return false;
        }

        return true;
    }

    void mpi_environment::init(int *argc, char ***argv, command_line_handling& cfg)
    {
        using namespace boost::assign;

        int this_rank = -1;
        has_called_init_ = false;

        // We assume to use the MPI parcelport if it is not explicitly disabled
        enabled_ = check_mpi_environment(cfg.rtcfg_);
        if (!enabled_)
        {
            cfg.ini_config_.push_back("hpx.parcel.mpi.enable = 0");
            return;
        }

        cfg.ini_config_ += "hpx.parcel.bootstrap!=mpi";

#if defined(HPX_HAVE_PARCELPORT_MPI_MULTITHREADED)
        int flag = (detail::get_cfg_entry(
            cfg, "hpx.parcel.mpi.multithreaded", 1) != 0) ?
                MPI_THREAD_MULTIPLE : MPI_THREAD_SINGLE;

#if defined(MVAPICH2_VERSION) && defined(_POSIX_SOURCE)
        // This enables multi threading support in MVAPICH2 if requested.
        if(flag == MPI_THREAD_MULTIPLE)
            setenv("MV2_ENABLE_AFFINITY", "0", 1);
#endif

        int retval = MPI_Init_thread(argc, argv, flag, &provided_threading_flag_);
#else
        int retval = MPI_Init(argc, argv);
        provided_threading_flag_ = MPI_THREAD_SINGLE;
#endif
        if (MPI_SUCCESS != retval)
        {
            if (MPI_ERR_OTHER != retval)
            {
                // explicitly disable mpi if not run by mpirun
                cfg.ini_config_.push_back("hpx.parcel.mpi.enable = 0");

                enabled_ = false;

                int msglen = 0;
                char message[MPI_MAX_ERROR_STRING+1];
                MPI_Error_string(retval, message, &msglen);
                message[msglen] = '\0';

                std::string msg("mpi_environment::init: MPI_Init_thread failed: ");
                msg = msg + message + ".";
                throw std::runtime_error(msg.c_str());
            }

            // somebody has already called MPI_Init before, we should be fine
            has_called_init_ = false;
        }
        else
        {
            has_called_init_ = true;
        }

        MPI_Comm_dup(MPI_COMM_WORLD, &communicator_);

        if (provided_threading_flag_ < MPI_THREAD_SERIALIZED)
        {
            // explicitly disable mpi if not run by mpirun
            cfg.ini_config_.push_back("hpx.parcel.mpi.multithreaded = 0");
        }

        if(provided_threading_flag_ == MPI_THREAD_FUNNELED)
        {
            enabled_ = false;
            has_called_init_ = false;
            throw std::runtime_error("mpi_environment::init: MPI_Init_thread: "
                "The underlying MPI implementation only supports "
                "MPI_THREAD_FUNNELED. This mode is not supported by HPX. Please "
                "pass -Ihpx.parcel.mpi.multithreaded=0 to explicitly disable MPI"
                " multithreading.");
        }

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

        cfg.ini_config_ += std::string("hpx.parcel.mpi.rank!=") +
            std::to_string(this_rank);
        cfg.ini_config_ += std::string("hpx.parcel.mpi.processorname!=") +
            get_processor_name();

        cfg.node_ = std::size_t(this_rank);
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
        if(enabled() && has_called_init())
        {
            MPI_Finalize();
        }
    }

    bool mpi_environment::enabled()
    {
        return enabled_;
    }

    bool mpi_environment::multi_threaded()
    {
        return provided_threading_flag_ >= MPI_THREAD_SERIALIZED;
    }

    bool mpi_environment::has_called_init()
    {
        return has_called_init_;
    }

    int mpi_environment::size()
    {
        int res(-1);
        if(enabled())
            MPI_Comm_size(communicator(), &res);
        return res;
    }

    int mpi_environment::rank()
    {
        int res(-1);
        if(enabled())
            MPI_Comm_rank(communicator(), &res);
        return res;
    }

    MPI_Comm& mpi_environment::communicator()
    {
        return communicator_;
    }

    mpi_environment::scoped_lock::scoped_lock()
    {
        if(!multi_threaded())
            mtx_.lock();
    }

    mpi_environment::scoped_lock::~scoped_lock()
    {
        if(!multi_threaded())
            mtx_.unlock();
    }

    void mpi_environment::scoped_lock::unlock()
    {
        if(!multi_threaded())
            mtx_.unlock();
    }

    mpi_environment::scoped_try_lock::scoped_try_lock()
      : locked(true)
    {
        if(!multi_threaded())
        {
            locked = mtx_.try_lock();
        }
    }

    mpi_environment::scoped_try_lock::~scoped_try_lock()
    {
        if(!multi_threaded() && locked)
            mtx_.unlock();
    }

    void mpi_environment::scoped_try_lock::unlock()
    {
        if(!multi_threaded() && locked)
        {
            locked = false;
            mtx_.unlock();
        }
    }
}}


//  Copyright (c) 2013-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_MPI_ENV_HPP
#define HPX_UTIL_MPI_ENV_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCELPORT_MPI)

#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util_fwd.hpp>

#include <mpi.h>

#include <cstdlib>
#include <string>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace util
{
    struct HPX_EXPORT mpi_environment
    {
        static void init(int *argc, char ***argv, command_line_handling& cfg);
        static void finalize();

        static bool enabled();
        static bool multi_threaded();
        static bool has_called_init();

        static int rank();
        static int size();

        static MPI_Comm& communicator();

        static std::string get_processor_name();

        static bool check_mpi_environment(runtime_configuration const& cfg);

        struct scoped_lock
        {
            scoped_lock();
            scoped_lock(scoped_lock const&) = delete;
            scoped_lock& operator=(scoped_lock const&) = delete;
            ~scoped_lock();
            void unlock();
        };

        struct scoped_try_lock
        {
            scoped_try_lock();
            scoped_try_lock(scoped_try_lock const&) = delete;
            scoped_try_lock& operator=(scoped_try_lock const&) = delete;
            ~scoped_try_lock();
            void unlock();
            bool locked;
        };

        typedef hpx::lcos::local::spinlock mutex_type;

    private:

        static mutex_type mtx_;

        static bool enabled_;
        static bool has_called_init_;
        static int provided_threading_flag_;
        static MPI_Comm communicator_;
    };
}}

#include <hpx/config/warnings_suffix.hpp>

#endif

#endif

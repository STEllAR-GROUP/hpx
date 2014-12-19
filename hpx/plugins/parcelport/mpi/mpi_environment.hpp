//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_MPI_ENV_HPP
#define HPX_UTIL_MPI_ENV_HPP

#include <hpx/config/defines.hpp>

#include <mpi.h>

#include <hpx/hpx_fwd.hpp>
#include <cstdlib>

namespace hpx { namespace util
{
    struct command_line_handling;

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

    private:
        static bool enabled_;
        static bool has_called_init_;
        static int provided_threading_flag_;
        static MPI_Comm communicator_;
    };
}}

#endif

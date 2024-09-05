//  Copyright (c) 2013-2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI)) ||      \
    defined(HPX_HAVE_MODULE_MPI_BASE)

#include <hpx/modules/runtime_configuration.hpp>
#include <hpx/mpi_base/mpi.hpp>
#include <hpx/synchronization/spinlock.hpp>

#include <cstdlib>
#include <string>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::util {

    struct HPX_CORE_EXPORT mpi_environment
    {
        static bool check_mpi_environment(runtime_configuration const& cfg);

        static int init(int* argc, char*** argv, int const minimal,
            int const required, int& provided);
        static void init(int* argc, char*** argv, runtime_configuration& cfg);
        static void finalize() noexcept;

        static bool enabled() noexcept;
        static bool multi_threaded() noexcept;
        static bool has_called_init() noexcept;

        static int rank() noexcept;
        static int size() noexcept;

        static MPI_Comm& communicator() noexcept;

        static std::string get_processor_name();

        struct HPX_CORE_EXPORT scoped_lock
        {
            scoped_lock();

            scoped_lock(scoped_lock const&) = delete;
            scoped_lock(scoped_lock&&) = delete;

            scoped_lock& operator=(scoped_lock const&) = delete;
            scoped_lock& operator=(scoped_lock&&) = delete;

            ~scoped_lock();

            constexpr bool owns_lock() const noexcept
            {
                return locked;
            }

            void unlock();
            bool locked;
        };

        struct HPX_CORE_EXPORT scoped_try_lock
        {
            scoped_try_lock();

            scoped_try_lock(scoped_try_lock const&) = delete;
            scoped_try_lock(scoped_try_lock&&) = delete;

            scoped_try_lock& operator=(scoped_try_lock const&) = delete;
            scoped_try_lock& operator=(scoped_try_lock&&) = delete;

            ~scoped_try_lock();

            constexpr bool owns_lock() const noexcept
            {
                return locked;
            }

            void unlock();
            bool locked;
        };

        using mutex_type = hpx::spinlock;

        static void check_mpi_error(
            scoped_lock& l, hpx::source_location const& sl, int error);
        static void check_mpi_error(
            scoped_try_lock& l, hpx::source_location const& sl, int error);

        // The highest order bit is used for acknowledgement messages
        static int MPI_MAX_TAG;

        constexpr static unsigned int MPI_ACK_MASK = 0xC000;
        constexpr static unsigned int MPI_ACK_TAG = 0x4000;

    private:
        static mutex_type mtx_;

        static bool enabled_;
        static bool has_called_init_;
        static int provided_threading_flag_;
        static MPI_Comm communicator_;

        static int is_initialized_;
    };
}    // namespace hpx::util

#include <hpx/config/warnings_suffix.hpp>

#else

#include <hpx/modules/runtime_configuration.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace util {
    struct HPX_CORE_EXPORT mpi_environment
    {
        static bool check_mpi_environment(runtime_configuration const& cfg);
    };
}}    // namespace hpx::util

#include <hpx/config/warnings_suffix.hpp>

#endif

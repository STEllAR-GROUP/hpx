//  Copyright (c) 2013-2015 Thomas Heller
//  Copyright (c) 2023 Tactical Computing Labs, LLC (Christopher Taylor)
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_GASNET)) ||      \
    defined(HPX_HAVE_MODULE_GASNET_BASE)

#include <hpx/gasnet_base/gasnet.hpp>
#include <hpx/modules/runtime_configuration.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/synchronization/mutex.hpp>

#include <cstdlib>
#include <string>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace util {

    struct HPX_CORE_EXPORT gasnet_environment
    {
        static bool check_gasnet_environment(runtime_configuration const& cfg);

        static int init(int* argc, char*** argv, const int minimal,
            const int required, int& provided);
        static void init(int* argc, char*** argv, runtime_configuration& cfg);
        static void finalize();

        static bool enabled();
        static bool multi_threaded();
        static bool has_called_init();

        static int rank();
        static int size();

        static std::string get_processor_name();

        static bool gettable(const int node, void* start, const size_t len);

        static void put(std::uint8_t * addr, const int rank, std::uint8_t * raddr, const std::size_t size);

        static void get(std::uint8_t * addr, const int rank, std::uint8_t * raddr, const std::size_t size);

        struct HPX_CORE_EXPORT scoped_lock
        {
            scoped_lock();
            scoped_lock(scoped_lock const&) = delete;
            scoped_lock& operator=(scoped_lock const&) = delete;
            ~scoped_lock();
            void unlock();
        };

        struct HPX_CORE_EXPORT scoped_try_lock
        {
            scoped_try_lock();
            scoped_try_lock(scoped_try_lock const&) = delete;
            scoped_try_lock& operator=(scoped_try_lock const&) = delete;
            ~scoped_try_lock();
            void unlock();
            bool locked;
        };

        typedef hpx::spinlock mutex_type;

    public:
        static hpx::spinlock pollingLock;
        static hpx::mutex mtx_;

        static bool enabled_;
        static bool has_called_init_;
        static int provided_threading_flag_;

        static int is_initialized_;

        static hpx::mutex dshm_mut;
        static int init_val_;
        static hpx::mutex* segment_mutex;
        static gasnet_seginfo_t* segments;
    };
}}    // namespace hpx::util

#include <hpx/config/warnings_suffix.hpp>

#else

#include <hpx/modules/runtime_configuration.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace util {
    struct HPX_CORE_EXPORT gasnet_environment
    {
        static bool check_gasnet_environment(runtime_configuration const& cfg);
    };
}}    // namespace hpx::util

#include <hpx/config/warnings_suffix.hpp>

#endif

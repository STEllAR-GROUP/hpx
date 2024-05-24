//  Copyright (c) 2013-2015 Thomas Heller
//  Copyright (c) 2023 Christopher Taylor
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_OPENSHMEM))

#include <hpx/modules/runtime_configuration.hpp>
#include <hpx/openshmem_base/openshmem.hpp>
#include <hpx/synchronization/mutex.hpp>
#include <hpx/synchronization/spinlock.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace util {

    struct openshmem_seginfo_t
    {
        std::uint8_t* beg_addr;
        std::uint8_t* end_addr;
        std::uint8_t * rcv;
        std::uint8_t * xmt;
//        std::shared_ptr<hpx::spinlock> * mut;

        openshmem_seginfo_t() : beg_addr(nullptr), end_addr(nullptr), rcv(nullptr), xmt(nullptr) {} //, mut(nullptr) {}
    };

    struct HPX_CORE_EXPORT openshmem_environment
    {
        static bool check_openshmem_environment(
            runtime_configuration const& cfg);

        static int init(int* argc, char*** argv, int& provided);
        static void init(int* argc, char*** argv, runtime_configuration& cfg);
        static void finalize();

        static bool enabled();
        static bool multi_threaded();
        static bool has_called_init();

        static int rank();
        static int size();

        static std::string get_processor_name();

	static void fence();
	static void quiet();

        static void put_signal(const std::uint8_t* addr, const int rank,
            std::uint8_t* raddr, const std::size_t size, std::uint8_t * sigaddr);

        static void wait_until(const std::uint8_t value, std::uint8_t * sigaddr);
        static std::size_t wait_until_any(const std::uint8_t value, std::uint8_t * sigaddr, const std::size_t count);

        static void get(std::uint8_t* addr, const int rank,
            const std::uint8_t* raddr, const std::size_t size);

        static void quiet() { shmem_quiet(); }

        static void global_barrier();

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

        typedef hpx::spinlock mutex_type;

    public:
        static hpx::spinlock mtx_;

        static bool enabled_;
        static bool has_called_init_;
        static int provided_threading_flag_;

        static int is_initialized_;

        static int init_val_;
        static std::vector<openshmem_seginfo_t> segments;
        static std::uint8_t* shmem_buffer;
    };
}}    // namespace hpx::util

#include <hpx/config/warnings_suffix.hpp>

#else

#include <hpx/modules/runtime_configuration.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace util {
    struct HPX_CORE_EXPORT openshmem_environment
    {
        static bool check_openshmem_environment(
            runtime_configuration const& cfg);
    };
}}    // namespace hpx::util

#include <hpx/config/warnings_suffix.hpp>

#endif

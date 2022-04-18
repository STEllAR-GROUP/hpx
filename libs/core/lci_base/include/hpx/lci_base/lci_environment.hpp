//  Copyright (c) 2013-2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)) ||      \
    defined(HPX_HAVE_MODULE_LCI_BASE)

#include <hpx/lci_base/lci.hpp>
#include <hpx/modules/runtime_configuration.hpp>
#include <hpx/synchronization/spinlock.hpp>

#include <cstdlib>
#include <string>
#include <thread>
#include <atomic>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace util {
    struct HPX_EXPORT lci_environment
    {
        static bool check_lci_environment(runtime_configuration const& cfg);

        static LCI_error_t init_lci();
        static void init(int* argc, char*** argv, runtime_configuration& cfg);
        static void finalize();

        static void progress_fn();

        static bool enabled();

        static int rank();
        static int size();

        static LCI_endpoint_t& lci_endpoint();

        static LCI_endpoint_t& rt_endpoint();

        static LCI_endpoint_t& h_endpoint();

        static LCI_comp_t& rt_queue();

        static LCI_comp_t& h_queue();

        static std::string get_processor_name();

        struct HPX_EXPORT scoped_lock
        {
            scoped_lock();
            scoped_lock(scoped_lock const&) = delete;
            scoped_lock& operator=(scoped_lock const&) = delete;
            ~scoped_lock();
            void unlock();
        };

        struct HPX_EXPORT scoped_try_lock
        {
            scoped_try_lock();
            scoped_try_lock(scoped_try_lock const&) = delete;
            scoped_try_lock& operator=(scoped_try_lock const&) = delete;
            ~scoped_try_lock();
            void unlock();
            bool locked;
        };

        typedef hpx::spinlock mutex_type;

    private:
        static mutex_type mtx_;
        static bool enabled_;
        static LCI_endpoint_t ep_;
        static LCI_endpoint_t rt_ep_;
        static LCI_endpoint_t h_ep_;
        static LCI_comp_t rt_cq_r_;
        static LCI_comp_t h_cq_r_;
        static std::thread* prg_thread_p;
        static std::atomic<bool> prg_thread_flag;
    };
}}    // namespace hpx::util

#include <hpx/config/warnings_suffix.hpp>

#else

#include <hpx/modules/runtime_configuration.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace util {
    struct HPX_EXPORT lci_environment
    {
        static bool check_lci_environment(runtime_configuration const& cfg);
    };
}}    // namespace hpx::util

#include <hpx/config/warnings_suffix.hpp>

#endif

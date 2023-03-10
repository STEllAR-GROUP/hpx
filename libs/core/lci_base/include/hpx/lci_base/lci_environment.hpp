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

#include <atomic>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace util {
    struct HPX_EXPORT lci_environment
    {
        static bool check_lci_environment(runtime_configuration& cfg);

        static void init_config(runtime_configuration& cfg);
        static void init(int* argc, char*** argv, runtime_configuration& cfg);
        static void setup(runtime_configuration& cfg);
        static void finalize();

        static void join_prg_thread_if_running();
        static void progress_fn(LCI_device_t device);
        static bool do_progress(LCI_device_t device);

        static bool enabled();

        static int rank();
        static int size();

        static LCI_device_t& get_device_eager();
        static LCI_device_t& get_device_iovec();

        static LCI_endpoint_t& get_endpoint_eager();
        static LCI_endpoint_t& get_endpoint_iovec();

        static LCI_comp_t& get_scq();

        static LCI_comp_t& get_rcq();

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
        static bool setuped_;
        static LCI_device_t device_eager;
        static LCI_device_t device_iovec;
        static LCI_endpoint_t ep_eager;
        static LCI_endpoint_t ep_iovec;
        static LCI_comp_t scq;
        static LCI_comp_t rcq;
        static std::unique_ptr<std::thread> prg_thread_eager_p;
        static std::unique_ptr<std::thread> prg_thread_iovec_p;
        static std::atomic<bool> prg_thread_flag;

    public:
        // configurations:
        // whether to use separate devices/progress threads for eager and iovec messages.
        static bool use_two_device;
        // whether to bypass the parcel queue and connection cache.
        static bool enable_send_immediate;
        // whether to use HPX resource partitioner to run the LCI progress function.
        static bool enable_lci_progress_pool;
        // whether to enable the backlog queue and eager message aggregation
        static bool enable_lci_backlog_queue;
    };
}}    // namespace hpx::util

#include <hpx/config/warnings_suffix.hpp>

#else

#include <hpx/modules/runtime_configuration.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace util {
    struct HPX_EXPORT lci_environment
    {
        static bool check_lci_environment(runtime_configuration& cfg);
    };
}}    // namespace hpx::util

#include <hpx/config/warnings_suffix.hpp>

#endif

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
        static void finalize();

        static bool do_progress(LCI_device_t device);
        static bool do_progress();

        static bool enabled();

        static int rank();
        static int size();

        static std::string get_processor_name();

        // log
        enum class log_level_t
        {
            none,
            profile,
            debug,
        };
        static log_level_t log_level;
#ifdef HPX_HAVE_PARCELPORT_LCI_LOG
        static LCT_log_ctx_t log_ctx;
#endif
        static void log(
            log_level_t level, const char* tag, const char* format, ...);
        // performance counter
        static LCT_pcounter_handle_t send_conn_start;
        static LCT_pcounter_handle_t send_conn_end;
        static LCT_pcounter_handle_t recv_conn_start;
        static LCT_pcounter_handle_t recv_conn_end;
        static LCT_pcounter_handle_t send_timer;
        static LCT_pcounter_handle_t handle_packet;
        static LCT_pcounter_ctx_t pcounter_ctx;
        static void pcounter_add(LCT_pcounter_handle_t handle, int64_t val);
        static void pcounter_start(LCT_pcounter_handle_t handle);
        static void pcounter_end(LCT_pcounter_handle_t handle);

    private:
        static bool enabled_;
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

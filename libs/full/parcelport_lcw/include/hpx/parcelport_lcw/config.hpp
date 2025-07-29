//  Copyright (c) 2023-2025 Jiakun Yan
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCW)

#include <hpx/modules/lcw_base.hpp>

namespace hpx::parcelset::policies::lcw {
    struct config_t
    {
        // whether init_config has been called
        static bool is_initialized;
        // whether to bypass the parcel queue and connection cache.
        static bool enable_send_immediate;
        // whether to enable the backlog queue and eager message aggregation
        static bool enable_lcw_backlog_queue;
        // how to run LCI_progress
        enum class progress_type_t
        {
            rp,                // HPX resource partitioner
            pthread,           // Normal progress pthread
            worker,            // HPX worker thread
            pthread_worker,    // Normal progress pthread + worker thread
        };
        static progress_type_t progress_type;
        // which device to make progress when a worker thread calls progress
        enum class progress_strategy_t
        {
            local,     // HPX resource partitioner
            global,    // Normal progress pthread
            random,    // HPX worker thread
        };
        static progress_strategy_t progress_strategy;
        // How many progress threads to create
        static int progress_thread_num;
        // How many devices to use
        static int ndevices;
        // How many completion managers to use
        static int ncomps;

        static void init_config(util::runtime_configuration const& rtcfg);
    };
}    // namespace hpx::parcelset::policies::lcw

#endif

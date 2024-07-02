//  Copyright (c) 2023-2024 Jiakun Yan
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)

#include <hpx/modules/lci_base.hpp>

namespace hpx::parcelset::policies::lci {
    struct config_t
    {
        // whether init_config has been called
        static bool is_initialized;
        // whether to bypass the parcel queue and connection cache.
        static bool enable_send_immediate;
        // whether to enable the backlog queue and eager message aggregation
        static bool enable_lci_backlog_queue;
        // which protocol to use
        enum class protocol_t
        {
            putva,
            sendrecv,
            putsendrecv,
        };
        static protocol_t protocol;
        // Whether sending header requires completion
        static bool enable_sendmc;
        // which completion mechanism to use for header messages
        enum class comp_type_t
        {
            queue,
            sync,
            sync_single,
            sync_single_nolock,
        };
        static comp_type_t completion_type_header;
        // which completion mechanism to use for followup messages
        static comp_type_t completion_type_followup;
        // how to run LCI_progress
        enum class progress_type_t
        {
            rp,                // HPX resource partitioner
            pthread,           // Normal progress pthread
            worker,            // HPX worker thread
            pthread_worker,    // Normal progress pthread + worker thread
            poll,              // progress when polling completion
        };
        static progress_type_t progress_type;
        // How many progress threads to create
        static int progress_thread_num;
        // How many pre-posted receives for new messages
        // (can only be applied to `sendrecv` protocol).
        static int prepost_recv_num;
        // Whether to register the buffer in HPX (or rely on LCI to register it)
        static bool reg_mem;
        // How many devices to use
        static int ndevices;
        // How many completion managers to use
        static int ncomps;
        // Whether to enable in-buffer assembly for the header messages.
        static bool enable_in_buffer_assembly;
        // The max retry count of send_nb before yield.
        static int send_nb_max_retry;
        // The max retry count of mbuffer_alloc before yield.
        static int mbuffer_alloc_max_retry;
        // The max count of background_work to invoke in a row
        static int bg_work_max_count;
        // Whether to do background work when sending
        static bool bg_work_when_send;

        static void init_config(util::runtime_configuration const& rtcfg);
    };
}    // namespace hpx::parcelset::policies::lci

#endif

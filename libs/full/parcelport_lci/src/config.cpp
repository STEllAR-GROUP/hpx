//  Copyright (c) 2022 Jiakun Yan
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)

#include <hpx/parcelport_lci/config.hpp>
#include <hpx/modules/lci_base.hpp>
#include <hpx/modules/runtime_configuration.hpp>
#include <hpx/modules/util.hpp>

#include <cstddef>
#include <string>

namespace hpx::parcelset::policies::lci {
    bool config_t::is_initialized = false;
    bool config_t::use_two_device;
    bool config_t::enable_send_immediate;
    bool config_t::enable_lci_backlog_queue;
    config_t::protocol_t config_t::protocol;
    LCI_comp_type_t config_t::completion_type;
    config_t::progress_type_t config_t::progress_type;
    int config_t::progress_thread_core;
    int config_t::prepost_recv_num;

    void config_t::init_config(util::runtime_configuration const& rtcfg)
    {
        if (is_initialized)
            return;
        is_initialized = true;
        // The default value here does not matter here
        enable_send_immediate = util::get_entry_as<bool>(
            rtcfg, "hpx.parcel.lci.sendimm", false /* Does not matter*/);
        enable_lci_backlog_queue = util::get_entry_as<bool>(
            rtcfg, "hpx.parcel.lci.backlog_queue", false /* Does not matter*/);
        use_two_device = util::get_entry_as<bool>(
            rtcfg, "hpx.parcel.lci.use_two_device", false);
        // set protocol to use
        std::string protocol_str = util::get_entry_as<std::string>(
            rtcfg, "hpx.parcel.lci.protocol", "");
        if (protocol_str == "putva")
        {
            protocol = protocol_t::putva;
        }
        else if (protocol_str == "putsendrecv")
        {
            protocol = protocol_t::putsendrecv;
        }
        else if (protocol_str == "sendrecv")
        {
            protocol = protocol_t::sendrecv;
        }
        else
        {
            throw std::runtime_error("Unknown protocol " + protocol_str);
        }
        // set completion mechanism to use
        std::string completion_str = util::get_entry_as<std::string>(
            rtcfg, "hpx.parcel.lci.comp_type", "");
        if (completion_str == "queue")
        {
            completion_type = LCI_COMPLETION_QUEUE;
        }
        else if (completion_str == "sync")
        {
            completion_type = LCI_COMPLETION_SYNC;
        }
        else
        {
            throw std::runtime_error(
                "Unknown completion type " + completion_str);
        }
        // set the way to run LCI_progress
        std::string progress_type_str = util::get_entry_as<std::string>(
            rtcfg, "hpx.parcel.lci.progress_type", "");
        if (progress_type_str == "rp")
        {
            progress_type = progress_type_t::rp;
        }
        else if (progress_type_str == "pthread")
        {
            progress_type = progress_type_t::pthread;
        }
        else if (progress_type_str == "worker")
        {
            progress_type = progress_type_t::worker;
        }
        else
        {
            throw std::runtime_error(
                "Unknown progress type " + progress_type_str);
        }
        progress_thread_core =
            util::get_entry_as(rtcfg, "hpx.parcel.lci.prg_thread_core", -1);
        prepost_recv_num =
            util::get_entry_as(rtcfg, "hpx.parcel.lci.prepost_recv_num", 1);

        if (protocol == protocol_t::sendrecv)
        {
            if (use_two_device)
            {
                use_two_device = false;
                fprintf(stderr, "WARNING: set use_two_device to false!\n");
            }
        }
        if (!enable_send_immediate && enable_lci_backlog_queue)
        {
            enable_lci_backlog_queue = false;
            fprintf(
                stderr, "WARNING: set enable_lci_backlog_queue to false!\n");
        }
        std::size_t num_threads =
            util::get_entry_as<size_t>(rtcfg, "hpx.os_threads", 1);
        if (progress_type == progress_type_t::rp && num_threads <= 1)
        {
            progress_type = progress_type_t::pthread;
            fprintf(stderr, "WARNING: set progress_type to pthread!\n");
        }
        if (use_two_device && progress_type == progress_type_t::rp &&
            num_threads <= 2)
        {
            use_two_device = false;
            fprintf(stderr, "WARNING: set use_two_device to false!\n");
        }
    }
}    // namespace hpx::parcelset::policies::lci
#endif

//  Copyright (c) 2023-2024 Jiakun Yan
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
    bool config_t::enable_send_immediate;
    bool config_t::enable_lci_backlog_queue;
    config_t::protocol_t config_t::protocol;
    config_t::comp_type_t config_t::completion_type_header;
    config_t::comp_type_t config_t::completion_type_followup;
    config_t::progress_type_t config_t::progress_type;
    int config_t::progress_thread_num;
    int config_t::prepost_recv_num;
    bool config_t::reg_mem;
    int config_t::ndevices;
    int config_t::ncomps;
    bool config_t::enable_in_buffer_assembly;
    int config_t::send_nb_max_retry;
    int config_t::mbuffer_alloc_max_retry;
    int config_t::bg_work_max_count;
    bool config_t::bg_work_when_send;
    bool config_t::enable_sendmc;

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
        if (completion_str != "deprecated")
        {
            fprintf(stderr, "hpx.parcel.lci.comp_type is deprecated!\n");
        }
        // set completion mechanism to use for header messages
        std::string header_completion_str = util::get_entry_as<std::string>(
            rtcfg, "hpx.parcel.lci.comp_type_header", "");
        if (header_completion_str == "queue")
        {
            completion_type_header = comp_type_t::queue;
        }
        else if (header_completion_str == "sync")
        {
            completion_type_header = comp_type_t::sync;
        }
        else if (header_completion_str == "sync_single")
        {
            completion_type_header = comp_type_t::sync_single;
        }
        else if (header_completion_str == "sync_single_nolock")
        {
            completion_type_header = comp_type_t::sync_single_nolock;
        }
        else
        {
            throw std::runtime_error(
                "Unknown completion type for header messages" +
                header_completion_str);
        }
        // set completion mechanism to use for follow-up messages
        std::string followup_completion_str = util::get_entry_as<std::string>(
            rtcfg, "hpx.parcel.lci.comp_type_followup", "");
        if (followup_completion_str == "queue")
        {
            completion_type_followup = comp_type_t::queue;
        }
        else if (followup_completion_str == "sync")
        {
            completion_type_followup = comp_type_t::sync;
        }
        else
        {
            throw std::runtime_error(
                "Unknown completion type for followup messages " +
                followup_completion_str);
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
        else if (progress_type_str == "pthread_worker")
        {
            progress_type = progress_type_t::pthread_worker;
        }
        else if (progress_type_str == "poll")
        {
            progress_type = progress_type_t::poll;
        }
        else
        {
            throw std::runtime_error(
                "Unknown progress type " + progress_type_str);
        }
        progress_thread_num = util::get_entry_as(
            rtcfg, "hpx.parcel.lci.prg_thread_num", -1 /* Does not matter*/);
        prepost_recv_num = util::get_entry_as(
            rtcfg, "hpx.parcel.lci.prepost_recv_num", 1 /* Does not matter*/);
        reg_mem = util::get_entry_as(
            rtcfg, "hpx.parcel.lci.reg_mem", 1 /* Does not matter*/);
        ndevices = util::get_entry_as(
            rtcfg, "hpx.parcel.lci.ndevices", 1 /* Does not matter*/);
        ncomps = util::get_entry_as(
            rtcfg, "hpx.parcel.lci.ncomps", 1 /* Does not matter*/);
        enable_in_buffer_assembly = util::get_entry_as(rtcfg,
            "hpx.parcel.lci.enable_in_buffer_assembly", 1 /* Does not matter*/);
        send_nb_max_retry = util::get_entry_as(
            rtcfg, "hpx.parcel.lci.send_nb_max_retry", 0 /* Does not matter*/);
        mbuffer_alloc_max_retry = util::get_entry_as(rtcfg,
            "hpx.parcel.lci.mbuffer_alloc_max_retry", 0 /* Does not matter*/);
        bg_work_max_count = util::get_entry_as(
            rtcfg, "hpx.parcel.lci.bg_work_max_count", 0 /* Does not matter*/);
        bg_work_when_send = util::get_entry_as(
            rtcfg, "hpx.parcel.lci.bg_work_when_send", 0 /* Does not matter*/);
        enable_sendmc = util::get_entry_as(
            rtcfg, "hpx.parcel.lci.enable_sendmc", 0 /* Does not matter*/);

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
#ifndef LCI_ENABLE_MULTITHREAD_PROGRESS
        if (progress_type == progress_type_t::worker ||
            progress_thread_num > ndevices)
        {
            fprintf(stderr,
                "WARNING: Thread-safe LCI_progress is needed "
                "but not enabled during compilation!\n");
        }
#endif
        if (ncomps > ndevices)
        {
            int old_ncomps = ncomps;
            ncomps = ndevices;
            fprintf(stderr,
                "WARNING: the number of completion managers (%d) "
                "cannot exceed the number of devices (%d). "
                "ncomps is adjusted accordingly (%d).\n",
                old_ncomps, ndevices, ncomps);
        }
        if (protocol != protocol_t::sendrecv &&
            completion_type_header != comp_type_t::queue)
        {
            fprintf(stderr,
                "WARNING: we have to use completion type `queue` "
                "for putsendrecv/putva protocol. comp_type_header "
                "is adjusted accordingly\n");
            completion_type_header = comp_type_t::queue;
        }
        if (protocol == protocol_t::sendrecv &&
            (prepost_recv_num > 1 || ndevices > ncomps) &&
            !(completion_type_header == comp_type_t::queue ||
                completion_type_header == comp_type_t::sync))
        {
            fprintf(stderr,
                "WARNING: we have to use completion type `queue` "
                "or `sync` for sendrecv protocol with more than "
                "one preposted recvs or devices sharing completion"
                "managers. comp_type_header "
                "is adjusted accordingly\n");
            completion_type_header = comp_type_t::queue;
        }
    }
}    // namespace hpx::parcelset::policies::lci
#endif

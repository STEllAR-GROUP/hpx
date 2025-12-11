//  Copyright (c) 2022-2025 Jiakun Yan
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCW)

#include <hpx/parcelport_lcw/config.hpp>
#include <hpx/modules/lcw_base.hpp>
#include <hpx/modules/runtime_configuration.hpp>
#include <hpx/modules/util.hpp>

#include <cstddef>
#include <string>

namespace hpx::parcelset::policies::lcw {
    bool config_t::is_initialized = false;
    bool config_t::enable_send_immediate;
    bool config_t::enable_lcw_backlog_queue;
    config_t::progress_type_t config_t::progress_type;
    config_t::progress_strategy_t config_t::progress_strategy;
    int config_t::progress_thread_num;
    int config_t::ndevices;
    int config_t::ncomps;

    void config_t::init_config(util::runtime_configuration const& rtcfg)
    {
        if (is_initialized)
            return;
        is_initialized = true;
        // The default value here does not matter here
        enable_send_immediate = util::get_entry_as<bool>(
            rtcfg, "hpx.parcel.lcw.sendimm", false /* Does not matter*/);
        enable_lcw_backlog_queue = util::get_entry_as<bool>(
            rtcfg, "hpx.parcel.lcw.backlog_queue", false /* Does not matter*/);
        // set the way to run LCI_progress
        std::string progress_type_str = util::get_entry_as<std::string>(
            rtcfg, "hpx.parcel.lcw.progress_type", "");
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
        else
        {
            throw std::runtime_error(
                "Unknown progress type " + progress_type_str);
        }
        // set the progress strategy
        std::string progress_strategy_str = util::get_entry_as<std::string>(
            rtcfg, "hpx.parcel.lcw.progress_strategy", "");
        if (progress_strategy_str == "local")
        {
            progress_strategy = progress_strategy_t::local;
        }
        else if (progress_strategy_str == "global")
        {
            progress_strategy = progress_strategy_t::global;
        }
        else if (progress_strategy_str == "random")
        {
            progress_strategy = progress_strategy_t::random;
        }
        else
        {
            throw std::runtime_error(
                "Unknown progress strategy " + progress_strategy_str);
        }
        progress_thread_num =
            util::get_entry_as(rtcfg, "hpx.parcel.lcw.prg_thread_num", -1);
        ndevices = util::get_entry_as(rtcfg, "hpx.parcel.lcw.ndevices", 1);
        ncomps = util::get_entry_as(rtcfg, "hpx.parcel.lcw.ncomps", 1);

        if (!enable_send_immediate && enable_lcw_backlog_queue)
        {
            enable_lcw_backlog_queue = false;
            fprintf(
                stderr, "WARNING: set enable_lcw_backlog_queue to false!\n");
        }
        std::size_t num_threads =
            util::get_entry_as<size_t>(rtcfg, "hpx.os_threads", 1);
        if (progress_type == progress_type_t::rp && num_threads <= 1)
        {
            progress_type = progress_type_t::pthread;
            fprintf(stderr, "WARNING: set progress_type to pthread!\n");
        }
        if (static_cast<size_t>(ndevices) > num_threads)
        {
            int old_ndevices = ndevices;
            ndevices = static_cast<int>(num_threads);
            fprintf(stderr,
                "WARNING: the number of devices (%d) "
                "cannot exceed the number of threads (%zu). "
                "ndevices is adjusted accordingly (%d).",
                old_ndevices, num_threads, ndevices);
        }
        if (ncomps > ndevices)
        {
            int old_ncomps = ncomps;
            ncomps = ndevices;
            fprintf(stderr,
                "WARNING: the number of completion managers (%d) "
                "cannot exceed the number of devices (%d). "
                "ncomps is adjusted accordingly (%d).",
                old_ncomps, ndevices, ncomps);
        }
    }
}    // namespace hpx::parcelset::policies::lcw
#endif

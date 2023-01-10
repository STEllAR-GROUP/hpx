//  Copyright (c) 2013-2015 Thomas Heller
//  Copyright (c)      2020 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <hpx/assert.hpp>
#include <hpx/modules/lci_base.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/modules/runtime_configuration.hpp>
#include <hpx/modules/string_util.hpp>
#include <hpx/modules/util.hpp>
#include <asio/ip/host_name.hpp>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util {

    namespace detail {

        bool detect_lci_environment(
            util::runtime_configuration const& cfg, char const* default_env)
        {
#if defined(__bgq__)
            // If running on BG/Q, we can safely assume to always run in an
            // LCI environment
            return true;
#else
            std::string lci_environment_strings =
                cfg.get_entry("hpx.parcel.lci.env", default_env);

            hpx::string_util::char_separator sep(";,: ");
            hpx::string_util::tokenizer tokens(lci_environment_strings, sep);
            for (auto const& tok : tokens)
            {
                char* env = std::getenv(tok.c_str());
                if (env)
                {
                    LBT_(debug)
                        << "Found LCI environment variable: " << tok << "="
                        << std::string(env) << ", enabling LCI support\n";
                    return true;
                }
            }

            LBT_(info) << "No known LCI environment variable found, disabling "
                          "LCI support\n";
            return false;
#endif
        }
    }    // namespace detail

    bool lci_environment::check_lci_environment(
        util::runtime_configuration const& cfg)
    {    // TODO: this should be returning false when we're using the MPI environment
#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)
        // We disable the LCI parcelport if any of these hold:
        //
        // - The parcelport is explicitly disabled
        // - The application is not run in an LCI environment
        // - The TCP parcelport is enabled and has higher priority
        // - The MPI parcelport is enabled and has higher priority
        if (get_entry_as(cfg, "hpx.parcel.lci.enable", 1) == 0 ||
            (get_entry_as(cfg, "hpx.parcel.tcp.enable", 1) &&
                (get_entry_as(cfg, "hpx.parcel.tcp.priority", 1) >
                    get_entry_as(cfg, "hpx.parcel.lci.priority", 0))) ||
            (get_entry_as(cfg, "hpx.parcel.mpi.enable", 1) &&
                (get_entry_as(cfg, "hpx.parcel.mpi.priority", 1) >
                    get_entry_as(cfg, "hpx.parcel.lci.priority", 0))))
        {
            LBT_(info) << "LCI support disabled via configuration settings\n";
            return false;
        }

        // log message was already generated
        return detail::detect_lci_environment(cfg, HPX_HAVE_PARCELPORT_LCI_ENV);
#elif defined(HPX_HAVE_MODULE_LCI_BASE)
        // if LCI futures are enabled while networking is off we need to
        // check whether we were run using mpirun
        return detail::detect_lci_environment(cfg, HPX_HAVE_PARCELPORT_LCI_ENV);
#else
        return false;
#endif
    }
}}    // namespace hpx::util

#if (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)) ||      \
    defined(HPX_HAVE_MODULE_LCI_BASE)

namespace hpx { namespace util {

    lci_environment::mutex_type lci_environment::mtx_;
    bool lci_environment::enabled_ = false;
    LCI_endpoint_t lci_environment::ep_;
    LCI_endpoint_t lci_environment::rt_ep_;
    LCI_comp_t lci_environment::rt_cq_r_;
    LCI_endpoint_t lci_environment::h_ep_;
    LCI_comp_t lci_environment::h_cq_r_;
    // We need this progress thread to send early parcels
    std::unique_ptr<std::thread> lci_environment::prg_thread_p = nullptr;
    std::atomic<bool> lci_environment::prg_thread_flag = false;
    hpx::spinlock prg_thread_mtx;

    ///////////////////////////////////////////////////////////////////////////
    LCI_error_t lci_environment::init_lci()
    {
        int lci_initialized = 0;
        LCI_initialized(&lci_initialized);
        if (!lci_initialized)
        {
            LCI_error_t lci_retval = LCI_initialize();
            if (lci_retval != LCI_OK)
                return lci_retval;
        }

        // create main endpoint for pt2pt msgs
        LCI_plist_t plist_;
        LCI_plist_create(&plist_);
        LCI_plist_set_comp_type(plist_, LCI_PORT_COMMAND, LCI_COMPLETION_SYNC);
        LCI_plist_set_comp_type(plist_, LCI_PORT_MESSAGE, LCI_COMPLETION_SYNC);
        LCI_endpoint_init(&ep_, LCI_UR_DEVICE, plist_);
        LCI_plist_free(&plist_);

        // set endpoint for release tag msgs
        rt_ep_ = LCI_UR_ENDPOINT;
        rt_cq_r_ = LCI_UR_CQ;

        // create endpoint for header msgs
        LCI_plist_t h_plist_;
        LCI_plist_create(&h_plist_);
        LCI_queue_create(LCI_UR_DEVICE, &h_cq_r_);
        LCI_plist_set_comp_type(
            h_plist_, LCI_PORT_MESSAGE, LCI_COMPLETION_SYNC);
        LCI_plist_set_comp_type(
            h_plist_, LCI_PORT_COMMAND, LCI_COMPLETION_SYNC);
        LCI_plist_set_default_comp(h_plist_, h_cq_r_);
        LCI_endpoint_init(&h_ep_, LCI_UR_DEVICE, h_plist_);
        LCI_plist_free(&h_plist_);
        // DEBUG("Rank %d: Init lci env", LCI_RANK);
        HPX_ASSERT(prg_thread_flag == false);
        HPX_ASSERT(prg_thread_p == nullptr);
        prg_thread_flag = true;
        prg_thread_p = std::make_unique<std::thread>(progress_fn);
        return LCI_OK;
    }

    ///////////////////////////////////////////////////////////////////////////
    void lci_environment::init(
        int*, char***, util::runtime_configuration& rtcfg)
    {
        if (enabled_)
            return;    // don't call twice

        int this_rank = -1;

        // We assume to use the LCI parcelport if it is not explicitly disabled
        enabled_ = check_lci_environment(rtcfg);
        if (!enabled_)
        {
            rtcfg.add_entry("hpx.parcel.lci.enable", "0");
            return;
        }

        rtcfg.add_entry("hpx.parcel.bootstrap", "lci");

        LCI_error_t retval = init_lci();
        if (LCI_OK != retval)
        {
            // explicitly disable lci if not run by mpirun
            rtcfg.add_entry("hpx.parcel.lci.enable", "0");
            enabled_ = false;
            throw std::runtime_error(
                "lci_environment::init: LCI_initialize failed");
        }

        this_rank = rank();

#if defined(HPX_HAVE_NETWORKING)
        if (this_rank == 0)
        {
            rtcfg.mode_ = hpx::runtime_mode::console;
        }
        else
        {
            rtcfg.mode_ = hpx::runtime_mode::worker;
        }
#elif defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
        rtcfg.mode_ = hpx::runtime_mode::console;
#else
        rtcfg.mode_ = hpx::runtime_mode::local;
#endif

        rtcfg.add_entry("hpx.parcel.lci.rank", std::to_string(this_rank));
    }

    std::string lci_environment::get_processor_name()
    {
        return asio::ip::host_name();
    }

    void lci_environment::finalize()
    {
        if (enabled())
        {
            // for some reasons, this code block can be entered twice when HPX exits
            int lci_init = 0;
            LCI_initialized(&lci_init);
            if (lci_init)
            {
                join_prg_thread_if_running();
                LCI_finalize();
            }
        }
    }

    void lci_environment::join_prg_thread_if_running()
    {
        if (prg_thread_p)
        {
            std::lock_guard<hpx::spinlock> l(prg_thread_mtx);
            if (prg_thread_p) {
                prg_thread_flag = false;
                prg_thread_p->join();
                prg_thread_p.reset();
            }
        }
    }

    void lci_environment::progress_fn()
    {
        while (prg_thread_flag)
        {
            LCI_progress(LCI_UR_DEVICE);
        }
    }

    bool lci_environment::do_progress()
    {
        LCI_error_t ret = LCI_progress(LCI_UR_DEVICE);
        HPX_ASSERT(ret == LCI_OK || ret == LCI_ERR_RETRY);
        return ret == LCI_OK;
    }

    bool lci_environment::enabled()
    {
        return enabled_;
    }

    int lci_environment::size()
    {
        int res(-1);
        if (enabled())
            res = LCI_NUM_PROCESSES;
        return res;
    }

    int lci_environment::rank()
    {
        int res(-1);
        if (enabled())
            res = LCI_RANK;
        return res;
    }

    LCI_endpoint_t& lci_environment::lci_endpoint()
    {
        return ep_;
    }

    LCI_endpoint_t& lci_environment::rt_endpoint()
    {
        return rt_ep_;
    }

    LCI_comp_t& lci_environment::rt_queue()
    {
        return rt_cq_r_;
    }

    LCI_endpoint_t& lci_environment::h_endpoint()
    {
        return h_ep_;
    }

    LCI_comp_t& lci_environment::h_queue()
    {
        return h_cq_r_;
    }

    lci_environment::scoped_lock::scoped_lock()
    {
        mtx_.lock();
    }

    lci_environment::scoped_lock::~scoped_lock()
    {
        mtx_.unlock();
    }

    void lci_environment::scoped_lock::unlock()
    {
        mtx_.unlock();
    }

    lci_environment::scoped_try_lock::scoped_try_lock()
      : locked(true)
    {
        locked = mtx_.try_lock();
    }

    lci_environment::scoped_try_lock::~scoped_try_lock()
    {
        if (locked)
            mtx_.unlock();
    }

    void lci_environment::scoped_try_lock::unlock()
    {
        if (!locked)
        {
            locked = false;
            mtx_.unlock();
        }
    }
}}    // namespace hpx::util

#endif

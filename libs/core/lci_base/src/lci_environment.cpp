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
#if !defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)
            return false;
#endif
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
                LBT_(info)
                    << "LCI support disabled via configuration settings\n";
                return false;
            }
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
        }
    }    // namespace detail

    bool lci_environment::check_lci_environment(
        util::runtime_configuration& cfg)
    {
        bool ret =
            detail::detect_lci_environment(cfg, HPX_HAVE_PARCELPORT_LCI_ENV);
        if (!ret)
        {
            cfg.add_entry("hpx.parcel.lci.enable", "0");
        }
        return ret;
    }
}}    // namespace hpx::util

#if (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)) ||      \
    defined(HPX_HAVE_MODULE_LCI_BASE)

namespace hpx { namespace util {

    lci_environment::mutex_type lci_environment::mtx_;
    bool lci_environment::enabled_ = false;
    bool lci_environment::setuped_ = false;
    LCI_device_t lci_environment::device_eager;
    LCI_device_t lci_environment::device_iovec;
    LCI_endpoint_t lci_environment::ep_eager;
    LCI_endpoint_t lci_environment::ep_iovec;
    LCI_comp_t lci_environment::scq;
    LCI_comp_t lci_environment::rcq;
    // We need this progress thread to send early parcels
    std::unique_ptr<std::thread> lci_environment::prg_thread_eager_p = nullptr;
    std::unique_ptr<std::thread> lci_environment::prg_thread_iovec_p = nullptr;
    std::atomic<bool> lci_environment::prg_thread_flag = false;

    bool lci_environment::use_two_device = false;
    bool lci_environment::enable_send_immediate = false;
    bool lci_environment::enable_lci_progress_pool = false;
    bool lci_environment::enable_lci_backlog_queue = false;

    ///////////////////////////////////////////////////////////////////////////
    void lci_environment::init_config(util::runtime_configuration& rtcfg)
    {
        enable_lci_progress_pool = hpx::util::get_entry_as<bool>(
            rtcfg, "hpx.parcel.lci.rp_prg_pool", false /* Does not matter*/);
        // The default value here does not matter here
        // the key "hpx.parcel.lci.sendimm" is guaranteed to exist
        enable_send_immediate = hpx::util::get_entry_as<bool>(
            rtcfg, "hpx.parcel.lci.sendimm", false /* Does not matter*/);
        enable_lci_backlog_queue = hpx::util::get_entry_as<bool>(
            rtcfg, "hpx.parcel.lci.backlog_queue", false /* Does not matter*/);
        use_two_device =
            get_entry_as(rtcfg, "hpx.parcel.lci.use_two_device", false);

        if (!enable_send_immediate && enable_lci_backlog_queue)
        {
            throw std::runtime_error(
                "Backlog queue must be used with send_immediate enabled");
        }
        std::size_t num_threads =
            hpx::util::get_entry_as<size_t>(rtcfg, "hpx.os_threads", 1);
        if (enable_lci_progress_pool && num_threads <= 1)
        {
            enable_lci_progress_pool = false;
            fprintf(
                stderr, "WARNING: set enable_lci_progress_pool to false!\n");
        }
        if (use_two_device && enable_lci_progress_pool && num_threads <= 2)
        {
            use_two_device = false;
            fprintf(stderr, "WARNING: set use_two_device to false!\n");
        }
    }

    void set_affinity(pthread_t pthread_handler, size_t target)
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(target, &cpuset);
        int rv =
            pthread_setaffinity_np(pthread_handler, sizeof(cpuset), &cpuset);
        if (rv != 0)
        {
            fprintf(stderr, "ERROR %d thread affinity didn't work.\n", rv);
            exit(1);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void lci_environment::init(
        int*, char***, util::runtime_configuration& rtcfg)
    {
        if (enabled_)
            return;    // don't call twice

        LCI_error_t retval;
        int lci_initialized = 0;
        LCI_initialized(&lci_initialized);
        if (!lci_initialized)
        {
            retval = LCI_initialize();
            if (LCI_OK != retval)
            {
                rtcfg.add_entry("hpx.parcel.lci.enable", "0");
                enabled_ = false;
                throw std::runtime_error(
                    "lci_environment::init: LCI_initialize failed");
            }
        }

        int this_rank = rank();

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

        rtcfg.add_entry("hpx.parcel.bootstrap", "lci");
        rtcfg.add_entry("hpx.parcel.lci.rank", std::to_string(this_rank));
        enabled_ = true;
    }

    void lci_environment::setup(util::runtime_configuration& rtcfg)
    {
        if (!enabled_)
            return;
        if (setuped_)
            return;    // don't call twice

        init_config(rtcfg);
        // create ep, scq, rcq
        device_eager = LCI_UR_DEVICE;
        if (use_two_device)
            LCI_device_init(&device_iovec);
        else
            device_iovec = LCI_UR_DEVICE;
        LCI_queue_create(LCI_UR_DEVICE /* no use */, &scq);
        LCI_queue_create(LCI_UR_DEVICE /* no use */, &rcq);
        LCI_plist_t plist_;
        LCI_plist_create(&plist_);
        LCI_plist_set_default_comp(plist_, rcq);
        LCI_plist_set_comp_type(plist_, LCI_PORT_COMMAND, LCI_COMPLETION_QUEUE);
        LCI_plist_set_comp_type(plist_, LCI_PORT_MESSAGE, LCI_COMPLETION_QUEUE);
        LCI_endpoint_init(&ep_eager, device_eager, plist_);
        LCI_endpoint_init(&ep_iovec, device_iovec, plist_);
        LCI_plist_free(&plist_);

        // create progress thread
        HPX_ASSERT(prg_thread_flag == false);
        HPX_ASSERT(prg_thread_eager_p == nullptr);
        HPX_ASSERT(prg_thread_iovec_p == nullptr);
        prg_thread_flag = true;
        prg_thread_eager_p =
            std::make_unique<std::thread>(progress_fn, get_device_eager());
        if (use_two_device)
            prg_thread_iovec_p =
                std::make_unique<std::thread>(progress_fn, get_device_iovec());
        int target = get_entry_as(rtcfg, "hpx.parcel.lci.prg_thread_core", -1);
        if (target >= 0)
        {
            set_affinity(prg_thread_eager_p->native_handle(), target);
            if (use_two_device)
                set_affinity(prg_thread_iovec_p->native_handle(), target + 1);
        }
        setuped_ = true;
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
                if (setuped_)
                {
                    join_prg_thread_if_running();
                    // create ep, scq, rcq
                    LCI_endpoint_free(&ep_iovec);
                    LCI_endpoint_free(&ep_eager);
                    LCI_queue_free(&scq);
                    LCI_queue_free(&rcq);
                    if (use_two_device)
                        LCI_device_free(&device_iovec);
                }
                LCI_finalize();
            }
        }
    }

    void lci_environment::join_prg_thread_if_running()
    {
        if (prg_thread_eager_p || prg_thread_iovec_p)
        {
            prg_thread_flag = false;
            if (prg_thread_eager_p)
            {
                prg_thread_eager_p->join();
                prg_thread_eager_p.reset();
            }
            if (prg_thread_iovec_p)
            {
                prg_thread_iovec_p->join();
                prg_thread_iovec_p.reset();
            }
        }
    }

    void lci_environment::progress_fn(LCI_device_t device)
    {
        while (prg_thread_flag)
        {
            LCI_progress(device);
        }
    }

    bool lci_environment::do_progress(LCI_device_t device)
    {
        LCI_error_t ret = LCI_progress(device);
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

    LCI_device_t& lci_environment::get_device_eager()
    {
        return device_eager;
    }

    LCI_device_t& lci_environment::get_device_iovec()
    {
        return device_iovec;
    }

    LCI_endpoint_t& lci_environment::get_endpoint_eager()
    {
        return ep_eager;
    }

    LCI_endpoint_t& lci_environment::get_endpoint_iovec()
    {
        return ep_iovec;
    }

    LCI_comp_t& lci_environment::get_scq()
    {
        return scq;
    }

    LCI_comp_t& lci_environment::get_rcq()
    {
        return rcq;
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

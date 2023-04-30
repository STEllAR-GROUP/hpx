//  Copyright (c) 2013-2015 Thomas Heller
//  Copyright (c)      2020 Google
//  Copyright (c)      2022 Patrick Diehl
//  Copyright (c)      2023 Tactical Computing Labs, LLC (Christopher Taylor)
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <hpx/modules/logging.hpp>
#include <hpx/modules/gasnet_base.hpp>
#include <hpx/modules/runtime_configuration.hpp>
#include <hpx/modules/util.hpp>

#include <boost/tokenizer.hpp>

#include <cstddef>
#include <cstdlib>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util {

    namespace detail {

        bool detect_gasnet_environment(
            util::runtime_configuration const& cfg, char const* default_env)
        {
            std::string gasnet_environment_strings =
                cfg.get_entry("hpx.parcel.gasnet.env", default_env);

            boost::char_separator<char> sep(";,: ");
            boost::tokenizer<boost::char_separator<char>> tokens(
                gasnet_environment_strings, sep);
            for (auto const& tok : tokens)
            {
                char* env = std::getenv(tok.c_str());
                if (env)
                {
                    LBT_(debug)
                        << "Found GASNET environment variable: " << tok << "="
                        << std::string(env) << ", enabling GASNET support\n";
                    return true;
                }
            }

            LBT_(info) << "No known GASNET environment variable found, disabling "
                          "GASNET support\n";
            return false;
        }
    }    // namespace detail

    bool gasnet_environment::check_gasnet_environment(
        util::runtime_configuration const& cfg)
    {
#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_GASNET)
        // We disable the GASNET parcelport if any of these hold:
        //
        // - The parcelport is explicitly disabled
        // - The application is not run in an GASNET environment
        // - The TCP parcelport is enabled and has higher priority
        if (get_entry_as(cfg, "hpx.parcel.gasnet.enable", 1) == 0 ||
            (get_entry_as(cfg, "hpx.parcel.udp.enable", 1) &&
                (get_entry_as(cfg, "hpx.parcel.udp.priority", 1) >
                    get_entry_as(cfg, "hpx.parcel.gasnet.priority", 0))) ||
            (get_entry_as(cfg, "hpx.parcel.gasnet.enable", 1) &&
                (get_entry_as(cfg, "hpx.parcel.gasnet.priority", 1) >
                    get_entry_as(cfg, "hpx.parcel.gasnet.priority", 0))))
        {
            LBT_(info) << "GASNET support disabled via configuration settings\n";
            return false;
        }

        return true;
#elif defined(HPX_HAVE_MODULE_GASNET_BASE)
        // if GASNET futures are enabled while networking is off we need to
        // check whether we were run using gasnetrun
        return detail::detect_gasnet_environment(cfg, HPX_HAVE_PARCELPORT_GASNET_ENV);
#else
        return false;
#endif
    }
}}    // namespace hpx::util

#if (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_GASNET)) ||      \
    defined(HPX_HAVE_MODULE_GASNET_BASE)

namespace hpx { namespace util {

    gasnet_environment::mutex_type gasnet_environment::mtx_;
    bool gasnet_environment::enabled_ = false;
    bool gasnet_environment::has_called_init_ = false;
    int gasnet_environment::provided_threading_flag_ = GASNET_SEQ;
    int gasnet_environment::is_initialized_ = -1;
    int gasnet_environment::init_val_ = GASNET_ERR_RESOURCE;

    ///////////////////////////////////////////////////////////////////////////
    int gasnet_environment::init(
        int* argc, char*** argv, const int minimal, const int required, int& provided)
    {
        if(!has_called_init_) {
           gasnet_environment::init_val_ = gasnet_init(argc, argv);
           has_called_init_ = true;
        }

        if(gasnet_environment::init_val_ == GASNET_ERR_NOT_INIT) {
           HPX_THROW_EXCEPTION(invalid_status,
               "hpx::util::gasnet_environment::init",
               "GASNET initialization error");
        }
        else if(gasnet_environment::init_val_ == GASNET_ERR_RESOURCE) {
           HPX_THROW_EXCEPTION(invalid_status,
               "hpx::util::gasnet_environment::init",
               "GASNET resource error");
        }
        else if(gasnet_environment::init_val_ == GASNET_ERR_BAD_ARG) {
           HPX_THROW_EXCEPTION(invalid_status,
               "hpx::util::gasnet_environment::init",
               "GASNET bad argument error");
        }
        else if(gasnet_environment::init_val_ == GASNET_ERR_NOT_READY) {
           HPX_THROW_EXCEPTION(invalid_status,
               "hpx::util::gasnet_environment::init",
               "GASNET not ready error");
        }

        if (provided < minimal) {
           HPX_THROW_EXCEPTION(invalid_status,
               "hpx::util::gasnet_environment::init",
               "GASNET doesn't provide minimal requested thread level");
        }

        return gasnet_environment::init_val_;
    }

    ///////////////////////////////////////////////////////////////////////////
    void gasnet_environment::init(
        int* argc, char*** argv, util::runtime_configuration& rtcfg)
    {
        if (enabled_)
            return;    // don't call twice

        int this_rank = -1;
        has_called_init_ = false;

        // We assume to use the GASNET parcelport if it is not explicitly disabled
        enabled_ = check_gasnet_environment(rtcfg);
        if (!enabled_)
        {
            rtcfg.add_entry("hpx.parcel.gasnet.enable", "0");
            return;
        }

        rtcfg.add_entry("hpx.parcel.bootstrap", "gasnet");

        int required = GASNET_SEQ;
        int retval =
            init(argc, argv, required, required, provided_threading_flag_);
        if (GASNET_OK != retval)
        {
            // explicitly disable gasnet if not run by gasnetrun
            rtcfg.add_entry("hpx.parcel.gasnet.enable", "0");

            enabled_ = false;

            int msglen = 0;
            char message[1024 + 1];
            sprintf(message, "%s\n", gasnet_ErrorDesc(retval));
            msglen = strnlen(message, 1025);

            std::string msg("gasnet_environment::init: gasnet_init failed: ");
            msg = msg + message + ".";
            throw std::runtime_error(msg.c_str());
        }

        if (provided_threading_flag_ != GASNET_SEQ)
        {
            // explicitly disable gasnet if not run by gasnetrun
            rtcfg.add_entry("hpx.parcel.gasnet.multithreaded", "0");
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

        rtcfg.add_entry("hpx.parcel.gasnet.rank", std::to_string(this_rank));
        rtcfg.add_entry("hpx.parcel.gasnet.processorname", get_processor_name());
    }

    std::string gasnet_environment::get_processor_name()
    {
        char name[1024 + 1] = {'\0'};
        const std::string rnkstr = std::to_string(rank());
        const int len = rnkstr.size();
        if(1025 < len) {
           HPX_THROW_EXCEPTION(invalid_status,
               "hpx::util::gasnet_environment::get_processor_name",
               "GASNET processor name is larger than 1025");
        }
        std::copy(std::begin(rnkstr), std::end(rnkstr), name);
        return name;
    }

    void gasnet_environment::finalize()
    {
        if (enabled() && has_called_init()) {
            gasnet_exit(1);
        }
    }

    bool gasnet_environment::enabled()
    {
        return enabled_;
    }

    bool gasnet_environment::multi_threaded()
    {
        return provided_threading_flag_ != GASNET_SEQ;
    }

    bool gasnet_environment::has_called_init()
    {
        return has_called_init_;
    }

    int gasnet_environment::size()
    {
        int res(-1);
        if (enabled())
            res = static_cast<int>(gasnet_nodes());
        return res;
    }

    int gasnet_environment::rank()
    {
        int res(-1);
        if (enabled())
            res = static_cast<int>(gasnet_mynode());
        return res;
    }

    gasnet_environment::scoped_lock::scoped_lock()
    {
        if (!multi_threaded())
            mtx_.lock();
    }

    gasnet_environment::scoped_lock::~scoped_lock()
    {
        if (!multi_threaded())
            mtx_.unlock();
    }

    void gasnet_environment::scoped_lock::unlock()
    {
        if (!multi_threaded())
            mtx_.unlock();
    }

    gasnet_environment::scoped_try_lock::scoped_try_lock()
      : locked(true)
    {
        if (!multi_threaded())
        {
            locked = mtx_.try_lock();
        }
    }

    gasnet_environment::scoped_try_lock::~scoped_try_lock()
    {
        if (!multi_threaded() && locked)
            mtx_.unlock();
    }

    void gasnet_environment::scoped_try_lock::unlock()
    {
        if (!multi_threaded() && locked)
        {
            locked = false;
            mtx_.unlock();
        }
    }
}}    // namespace hpx::util

#endif

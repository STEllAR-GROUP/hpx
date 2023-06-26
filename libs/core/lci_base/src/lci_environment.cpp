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
#include <stdarg.h>
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

    bool lci_environment::enabled_ = false;
    lci_environment::log_level_t lci_environment::log_level = log_level_t::none;
    FILE* lci_environment::log_outfile = nullptr;

    ///////////////////////////////////////////////////////////////////////////
    void lci_environment::init_config(util::runtime_configuration& rtcfg)
    {
        // The default value here does not matter here
        std::string log_level_str = get_entry_as<std::string>(
            rtcfg, "hpx.parcel.lci.log_level", "" /* Does not matter*/);
        if (log_level_str == "none")
            log_level = log_level_t::none;
        else if (log_level_str == "profile")
            log_level = log_level_t::profile;
        else if (log_level_str == "debug")
            log_level = log_level_t::debug;
        else
            throw std::runtime_error("Unknown log level " + log_level_str);
        std::string log_filename = get_entry_as<std::string>(
            rtcfg, "hpx.parcel.lci.log_outfile", "" /* Does not matter*/);
        if (log_filename == "stderr")
            log_outfile = stderr;
        else if (log_filename == "stdout")
            log_outfile = stdout;
        else
        {
            const int filename_max = 256;
            char filename[filename_max];
            char* p0_old = log_filename.data();
            char* p0_new = strchr(log_filename.data(), '%');
            char* p1 = filename;
            while (p0_new)
            {
                long nbytes = p0_new - p0_old;
                HPX_ASSERT(p1 + nbytes < filename + filename_max);
                memcpy(p1, p0_old, nbytes);
                p1 += nbytes;
                nbytes =
                    snprintf(p1, filename + filename_max - p1, "%d", LCI_RANK);
                p1 += nbytes;
                p0_old = p0_new + 1;
                p0_new = strchr(p0_old, '%');
            }
            strncat(p1, p0_old, filename + filename_max - p1 - 1);
            log_outfile = fopen(filename, "w+");
            if (log_outfile == nullptr)
            {
                throw std::runtime_error(
                    "Cannot open the logfile " + std::string(filename));
            }
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
        init_config(rtcfg);
        enabled_ = true;
    }

    std::string lci_environment::get_processor_name()
    {
        return asio::ip::host_name();
    }

    void lci_environment::finalize()
    {
        if (enabled())
        {
            enabled_ = false;
            // for some reasons, this code block can be entered twice when HPX exits
            int lci_init = 0;
            LCI_initialized(&lci_init);
            if (lci_init)
            {
                LCI_finalize();
            }
        }
    }

    bool lci_environment::do_progress(LCI_device_t device)
    {
        if (!device)
            return false;
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

    void lci_environment::log(
        lci_environment::log_level_t level, const char* format, ...)
    {
        va_list args;
        va_start(args, format);

        if (level <= log_level)
            vfprintf(log_outfile, format, args);

        va_end(args);
    }
}}    // namespace hpx::util

#endif

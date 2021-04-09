//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_LOGGING)
#include <hpx/assert.hpp>
#include <hpx/init_runtime_local/detail/init_logging.hpp>

#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util {
    void init_agas_console_log(util::section const& ini)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.console.agas"))
        {
            util::section const* logini =
                ini.get_section("hpx.logging.console.agas");
            HPX_ASSERT(nullptr != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            if (!loglevel.empty())
            {
                logdest = logini->get_entry("destination", empty);
                logformat =
                    detail::unescape(logini->get_entry("format", empty));
            }
        }

        auto lvl = hpx::util::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel, true);

        if (hpx::util::logging::level::disable_all != lvl)
        {
            logger_writer_type& writer = agas_console_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())    // ensure minimal defaults
                logdest = "android_log";
            writer.set_destination("android_log", android_log("hpx.agas"));
#else
            if (logdest.empty())    // ensure minimal defaults
                logdest = "cerr";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            writer.write(logformat, logdest);

            agas_console_logger()->mark_as_initialized();
            agas_console_logger()->set_enabled(lvl);
        }
    }

    void init_parcel_console_log(util::section const& ini)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.console.parcel"))
        {
            util::section const* logini =
                ini.get_section("hpx.logging.console.parcel");
            HPX_ASSERT(nullptr != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            if (!loglevel.empty())
            {
                logdest = logini->get_entry("destination", empty);
                logformat =
                    detail::unescape(logini->get_entry("format", empty));
            }
        }

        auto lvl = hpx::util::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel, true);

        if (hpx::util::logging::level::disable_all != lvl)
        {
            logger_writer_type& writer = parcel_console_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())    // ensure minimal defaults
                logdest = "android_log";
            writer.set_destination("android_log", android_log("hpx.parcel"));
#else
            if (logdest.empty())    // ensure minimal defaults
                logdest = "cerr";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            writer.write(logformat, logdest);

            parcel_console_logger()->mark_as_initialized();
            parcel_console_logger()->set_enabled(lvl);
        }
    }

    void init_timing_console_log(util::section const& ini)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.console.timing"))
        {
            util::section const* logini =
                ini.get_section("hpx.logging.console.timing");
            HPX_ASSERT(nullptr != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            if (!loglevel.empty())
            {
                logdest = logini->get_entry("destination", empty);
                logformat =
                    detail::unescape(logini->get_entry("format", empty));
            }
        }

        auto lvl = hpx::util::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel, true);

        if (hpx::util::logging::level::disable_all != lvl)
        {
            logger_writer_type& writer = timing_console_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())    // ensure minimal defaults
                logdest = "android_log";
            writer.set_destination("android_log", android_log("hpx.timing"));
#else
            if (logdest.empty())    // ensure minimal defaults
                logdest = "cerr";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            writer.write(logformat, logdest);

            timing_console_logger()->mark_as_initialized();
            timing_console_logger()->set_enabled(lvl);
        }
    }

    void init_hpx_console_log(util::section const& ini)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.console"))
        {
            util::section const* logini =
                ini.get_section("hpx.logging.console");
            HPX_ASSERT(nullptr != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            if (!loglevel.empty())
            {
                logdest = logini->get_entry("destination", empty);
                logformat =
                    detail::unescape(logini->get_entry("format", empty));
            }
        }

        auto lvl = hpx::util::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel, true);

        if (hpx::util::logging::level::disable_all != lvl)
        {
            logger_writer_type& writer = hpx_console_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())    // ensure minimal defaults
                logdest = "android_log";
            writer.set_destination("android_log", android_log("hpx"));
#else
            if (logdest.empty())    // ensure minimal defaults
                logdest = "cerr";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            writer.write(logformat, logdest);

            hpx_console_logger()->mark_as_initialized();
            hpx_console_logger()->set_enabled(lvl);
        }
    }

    void init_app_console_log(util::section const& ini)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.console.application"))
        {
            util::section const* logini =
                ini.get_section("hpx.logging.console.application");
            HPX_ASSERT(nullptr != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            if (!loglevel.empty())
            {
                logdest = logini->get_entry("destination", empty);
                logformat =
                    detail::unescape(logini->get_entry("format", empty));
            }
        }

        auto lvl = hpx::util::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel, true);

        if (hpx::util::logging::level::disable_all != lvl)
        {
            logger_writer_type& writer = app_console_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())    // ensure minimal defaults
                logdest = "android_log";
            writer.set_destination(
                "android_log", android_log("hpx.application"));
#else
            if (logdest.empty())    // ensure minimal defaults
                logdest = "cerr";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            writer.write(logformat, logdest);

            app_console_logger()->mark_as_initialized();
            app_console_logger()->set_enabled(lvl);
        }
    }

    void init_debuglog_console_log(util::section const& ini)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.console.debuglog"))
        {
            util::section const* logini =
                ini.get_section("hpx.logging.console.debuglog");
            HPX_ASSERT(nullptr != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            if (!loglevel.empty())
            {
                logdest = logini->get_entry("destination", empty);
                logformat =
                    detail::unescape(logini->get_entry("format", empty));
            }
        }

        auto lvl = hpx::util::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel, true);

        if (hpx::util::logging::level::disable_all != lvl)
        {
            logger_writer_type& writer = debuglog_console_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())    // ensure minimal defaults
                logdest = "android_log";
            writer.set_destination("android_log", android_log("hpx.debuglog"));
#else
            if (logdest.empty())    // ensure minimal defaults
                logdest = "cerr";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            writer.write(logformat, logdest);

            debuglog_console_logger()->mark_as_initialized();
            debuglog_console_logger()->set_enabled(lvl);
        }
    }
}}    // namespace hpx::util

#endif    // HPX_HAVE_LOGGING

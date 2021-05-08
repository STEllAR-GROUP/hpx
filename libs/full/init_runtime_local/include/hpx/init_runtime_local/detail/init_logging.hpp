//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_LOGGING)
#include <hpx/logging/format/named_write.hpp>
#include <hpx/logging/manipulator.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/modules/threading_base.hpp>
#include <hpx/runtime_configuration/runtime_configuration.hpp>
#include <hpx/runtime_local/get_locality_id.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/type_support/static.hpp>
#include <hpx/util/get_entry_as.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <string>
#include <vector>

#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#endif

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(push)
#pragma warning(                                                               \
    disable : 4250)    // 'class1' : inherits 'class2::member' via dominance
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util {
    using logger_writer_type = logging::writer::named_write;

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: shepherd
    struct shepherd_thread_id : logging::formatter::manipulator
    {
        shepherd_thread_id() {}

        void operator()(std::ostream& to) const override
        {
            error_code ec(lightweight);
            std::size_t thread_num = hpx::get_worker_thread_num(ec);

            if (std::size_t(-1) != thread_num)
            {
                util::format_to(to, "{:016x}", thread_num);
            }
            else
            {
                to << std::string(16, '-');
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: locality prefix
    struct locality_prefix : logging::formatter::manipulator
    {
        locality_prefix() {}

        void operator()(std::ostream& to) const override
        {
            std::uint32_t locality_id = hpx::get_locality_id();

            if (~static_cast<std::uint32_t>(0) != locality_id)
            {
                util::format_to(to, "{:08x}", locality_id);
            }
            else
            {
                // called from outside a HPX thread
                to << std::string(8, '-');
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: HPX thread id
    struct thread_id : logging::formatter::manipulator
    {
        void operator()(std::ostream& to) const override
        {
            threads::thread_self* self = threads::get_self_ptr();
            if (nullptr != self)
            {
                // called from inside a HPX thread
                threads::thread_id_type id = threads::get_self_id();
                if (id != threads::invalid_thread_id)
                {
                    std::ptrdiff_t value =
                        reinterpret_cast<std::ptrdiff_t>(id.get());
                    util::format_to(to, "{:016x}", value);
                    return;
                }
            }

            // called from outside a HPX thread or invalid thread id
            to << std::string(16, '-');
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: HPX thread phase
    struct thread_phase : logging::formatter::manipulator
    {
        void operator()(std::ostream& to) const override
        {
            threads::thread_self* self = threads::get_self_ptr();
            if (nullptr != self)
            {
                // called from inside a HPX thread
                std::size_t phase = self->get_thread_phase();
                if (0 != phase)
                {
                    util::format_to(to, "{:04x}", self->get_thread_phase());
                    return;
                }
            }

            // called from outside a HPX thread or no phase given
            to << std::string(4, '-');
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: locality prefix of parent thread
    struct parent_thread_locality : logging::formatter::manipulator
    {
        void operator()(std::ostream& to) const override
        {
            std::uint32_t parent_locality_id =
                threads::get_parent_locality_id();
            if (~static_cast<std::uint32_t>(0) != parent_locality_id)
            {
                // called from inside a HPX thread
                util::format_to(to, "{:08x}", parent_locality_id);
            }
            else
            {
                // called from outside a HPX thread
                to << std::string(8, '-');
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: HPX parent thread id
    struct parent_thread_id : logging::formatter::manipulator
    {
        void operator()(std::ostream& to) const override
        {
            threads::thread_id_type parent_id = threads::get_parent_id();
            if (nullptr != parent_id && threads::invalid_thread_id != parent_id)
            {
                // called from inside a HPX thread
                std::ptrdiff_t value =
                    reinterpret_cast<std::ptrdiff_t>(parent_id.get());
                util::format_to(to, "{:016x}", value);
            }
            else
            {
                // called from outside a HPX thread
                to << std::string(16, '-');
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: HPX parent thread phase
    struct parent_thread_phase : logging::formatter::manipulator
    {
        void operator()(std::ostream& to) const override
        {
            std::size_t parent_phase = threads::get_parent_phase();
            if (0 != parent_phase)
            {
                // called from inside a HPX thread
                util::format_to(to, "{:04x}", parent_phase);
            }
            else
            {
                // called from outside a HPX thread
                to << std::string(4, '-');
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom log destination: send generated strings to console
    struct console_local : logging::destination::manipulator
    {
        console_local(logging::level level, logging_destination dest)
          : level_(level)
          , dest_(dest)
        {
        }

        void operator()(logging::message const& msg) override
        {
            switch (dest_)
            {
            default:
            case destination_hpx:
                LHPX_CONSOLE_(level_) << msg;
                break;

            case destination_timing:
                LTIM_CONSOLE_(level_) << msg;
                break;

            case destination_agas:
                LAGAS_CONSOLE_(level_) << msg;
                break;

            case destination_parcel:
                LPT_CONSOLE_(level_) << msg;
                break;

            case destination_app:
                LAPP_CONSOLE_(level_) << msg;
                break;

            case destination_debuglog:
                LDEB_CONSOLE_ << msg;
                break;
            }
        }

        bool operator==(console_local const& rhs) const
        {
            return dest_ == rhs.dest_;
        }

        logging::level level_;
        logging_destination dest_;
    };    // namespace util

#if defined(ANDROID) || defined(__ANDROID__)
    // default log destination for Android
    struct android_log : logging::destination::manipulator
    {
        android_log(char const* tag_)
          : tag(tag_)
        {
        }

        void operator()(logging::message const& msg) override
        {
            __android_log_write(
                ANDROID_LOG_DEBUG, tag.c_str(), msg.full_string().c_str());
        }

        bool operator==(android_log const& rhs) const
        {
            return tag == rhs.tag;
        }

        std::string tag;
    };
#endif

    namespace detail {
        // unescape config entry
        inline std::string unescape(std::string const& value)
        {
            std::string result;
            std::string::size_type pos = 0;
            std::string::size_type pos1 = value.find_first_of('\\', 0);
            if (std::string::npos != pos1)
            {
                do
                {
                    switch (value[pos1 + 1])
                    {
                    case '\\':
                    case '\"':
                    case '?':
                        result = result + value.substr(pos, pos1 - pos);
                        pos1 = value.find_first_of('\\', (pos = pos1 + 1) + 1);
                        break;

                    case 'n':
                        result = result + value.substr(pos, pos1 - pos) + "\n";
                        pos1 = value.find_first_of('\\', pos = pos1 + 1);
                        ++pos;
                        break;

                    default:
                        result = result + value.substr(pos, pos1 - pos + 1);
                        pos1 = value.find_first_of('\\', pos = pos1 + 1);
                    }

                } while (pos1 != std::string::npos);
                result = result + value.substr(pos);
            }
            else
            {
                // the string doesn't contain any escaped character sequences
                result = value;
            }
            return result;
        }

        inline void define_formatters_local(logger_writer_type& writer)
        {
            writer.set_formatter("osthread", shepherd_thread_id());
            writer.set_formatter("locality", locality_prefix());
            writer.set_formatter("hpxthread", thread_id());
            writer.set_formatter("hpxphase", thread_phase());
            writer.set_formatter("hpxparent", parent_thread_id());
            writer.set_formatter("hpxparentphase", parent_thread_phase());
            writer.set_formatter("parentloc", parent_thread_locality());
        }
    }    // namespace detail

    // initialize logging for AGAS
    template <typename ConsoleDestination, typename DefineFormatters>
    void init_agas_log(util::section const& ini, bool isconsole,
        DefineFormatters&& define_formatters)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.agas"))
        {
            util::section const* logini = ini.get_section("hpx.logging.agas");
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
            lvl = detail::get_log_level(loglevel);

        if (hpx::util::logging::level::disable_all != lvl)
        {
            logger_writer_type& writer = agas_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())    // ensure minimal defaults
                logdest = isconsole ? "android_log" : "console";
            agas_logger()->writer().set_destination(
                "android_log", android_log("hpx.agas"));
#else
            if (logdest.empty())    // ensure minimal defaults
                logdest = isconsole ? "cerr" : "console";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            writer.set_destination("console",
                ConsoleDestination(lvl, destination_agas));    //-V106
            writer.write(logformat, logdest);
            define_formatters(writer);

            agas_logger()->mark_as_initialized();
            agas_logger()->set_enabled(lvl);
        }
    }

    // initialize logging for the parcel transport
    template <typename ConsoleDestination, typename DefineFormatters>
    void init_parcel_log(util::section const& ini, bool isconsole,
        DefineFormatters&& define_formatters)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.parcel"))
        {
            util::section const* logini = ini.get_section("hpx.logging.parcel");
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
            lvl = detail::get_log_level(loglevel);

        if (hpx::util::logging::level::disable_all != lvl)
        {
            logger_writer_type& writer = parcel_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())    // ensure minimal defaults
                logdest = isconsole ? "android_log" : "console";
            parcel_logger()->writer().set_destination(
                "android_log", android_log("hpx.parcel"));
#else
            if (logdest.empty())    // ensure minimal defaults
                logdest = isconsole ? "cerr" : "console";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            writer.set_destination("console",
                ConsoleDestination(lvl, destination_parcel));    //-V106
            writer.write(logformat, logdest);
            define_formatters(writer);

            parcel_logger()->mark_as_initialized();
            parcel_logger()->set_enabled(lvl);
        }
    }

    // initialize logging for performance measurements
    template <typename ConsoleDestination, typename DefineFormatters>
    void init_timing_log(util::section const& ini, bool isconsole,
        DefineFormatters&& define_formatters)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.timing"))
        {
            util::section const* logini = ini.get_section("hpx.logging.timing");
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
            lvl = detail::get_log_level(loglevel);

        if (hpx::util::logging::level::disable_all != lvl)
        {
            logger_writer_type& writer = timing_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())    // ensure minimal defaults
                logdest = isconsole ? "android_log" : "console";

            writer.set_destination("android_log", android_log("hpx.timing"));
#else
            if (logdest.empty())    // ensure minimal defaults
                logdest = isconsole ? "cerr" : "console";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            writer.set_destination("console",
                ConsoleDestination(lvl, destination_timing));    //-V106
            writer.write(logformat, logdest);
            define_formatters(writer);

            timing_logger()->mark_as_initialized();
            timing_logger()->set_enabled(lvl);
        }
    }

    template <typename ConsoleDestination, typename DefineFormatters>
    void init_hpx_logs(util::section const& ini, bool isconsole,
        DefineFormatters&& define_formatters)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging"))
        {
            util::section const* logini = ini.get_section("hpx.logging");
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

        logger_writer_type& writer = hpx_logger()->writer();
        logger_writer_type& error_writer = hpx_error_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
        if (logdest.empty())    // ensure minimal defaults
            logdest = isconsole ? "android_log" : "console";

        writer.set_destination("android_log", android_log("hpx"));
        error_writer.set_destination("android_log", android_log("hpx"));
#else
        if (logdest.empty())    // ensure minimal defaults
            logdest = isconsole ? "cerr" : "console";
#endif
        if (logformat.empty())
            logformat = "|\\n";

        if (hpx::util::logging::level::disable_all != lvl)
        {
            writer.set_destination(
                "console", ConsoleDestination(lvl, destination_hpx));    //-V106
            writer.write(logformat, logdest);
            define_formatters(writer);

            hpx_logger()->mark_as_initialized();
            hpx_logger()->set_enabled(lvl);

            // errors are logged to the given destination and to cerr
            error_writer.set_destination(
                "console", ConsoleDestination(lvl, destination_hpx));    //-V106
#if !defined(ANDROID) && !defined(__ANDROID__)
            if (logdest != "cerr")
                error_writer.write(logformat, logdest + " cerr");
#endif
            define_formatters(error_writer);

            hpx_error_logger()->mark_as_initialized();
            hpx_error_logger()->set_enabled(lvl);
        }
        else
        {
            // errors are always logged to cerr
            if (!isconsole)
            {
                error_writer.set_destination("console",
                    ConsoleDestination(lvl, destination_hpx));    //-V106
                error_writer.write(logformat, "console");
            }
            else
            {
#if defined(ANDROID) || defined(__ANDROID__)
                error_writer.write(logformat, "android_log");
#else
                error_writer.write(logformat, "cerr");
#endif
            }
            define_formatters(error_writer);

            hpx_error_logger()->mark_as_initialized();
            hpx_error_logger()->set_enabled(hpx::util::logging::level::fatal);
        }
    }

    // initialize logging for application
    template <typename ConsoleDestination, typename DefineFormatters>
    void init_app_logs(util::section const& ini, bool isconsole,
        DefineFormatters&& define_formatters)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.application"))
        {
            util::section const* logini =
                ini.get_section("hpx.logging.application");
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
            lvl = detail::get_log_level(loglevel);

        if (hpx::util::logging::level::disable_all != lvl)
        {
            logger_writer_type& writer = app_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())    // ensure minimal defaults
                logdest = isconsole ? "android_log" : "console";
            writer.set_destination(
                "android_log", android_log("hpx.application"));
#else
            if (logdest.empty())    // ensure minimal defaults
                logdest = isconsole ? "cerr" : "console";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            writer.set_destination(
                "console", ConsoleDestination(lvl, destination_app));    //-V106
            writer.write(logformat, logdest);
            define_formatters(writer);

            app_logger()->mark_as_initialized();
            app_logger()->set_enabled(lvl);
        }
    }

    // initialize logging for application
    template <typename ConsoleDestination, typename DefineFormatters>
    void init_debuglog_logs(util::section const& ini, bool isconsole,
        DefineFormatters&& define_formatters)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.debuglog"))
        {
            util::section const* logini =
                ini.get_section("hpx.logging.debuglog");
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
            lvl = detail::get_log_level(loglevel);

        if (hpx::util::logging::level::disable_all != lvl)
        {
            logger_writer_type& writer = debuglog_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())    // ensure minimal defaults
                logdest = isconsole ? "android_log" : "console";
            writer.set_destination("android_log", android_log("hpx.debuglog"));
#else
            if (logdest.empty())    // ensure minimal defaults
                logdest = isconsole ? "cerr" : "console";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            writer.set_destination("console",
                ConsoleDestination(lvl, destination_debuglog));    //-V106
            writer.write(logformat, logdest);
            define_formatters(writer);

            debuglog_logger()->mark_as_initialized();
            debuglog_logger()->set_enabled(lvl);
        }
    }

    // initialize logging for AGAS
    void init_agas_console_log(util::section const& ini);
    // initialize logging for the parcel transport
    void init_parcel_console_log(util::section const& ini);
    // initialize logging for performance measurements
    void init_timing_console_log(util::section const& ini);
    // initialize logging for HPX runtime
    void init_hpx_console_log(util::section const& ini);
    // initialize logging for applications
    void init_app_console_log(util::section const& ini);
    // initialize logging for applications
    void init_debuglog_console_log(util::section const& ini);
}}    // namespace hpx::util

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    template <typename ConsoleDestination, typename DefineFormatters>
    void init_logging(runtime_configuration& ini, bool isconsole,
        DefineFormatters&& define_formatters)
    {
        // initialize normal logs
        init_agas_log<ConsoleDestination>(ini, isconsole, define_formatters);
        init_parcel_log<ConsoleDestination>(ini, isconsole, define_formatters);
        init_timing_log<ConsoleDestination>(ini, isconsole, define_formatters);
        init_hpx_logs<ConsoleDestination>(ini, isconsole, define_formatters);
        init_app_logs<ConsoleDestination>(ini, isconsole, define_formatters);
        init_debuglog_logs<ConsoleDestination>(
            ini, isconsole, define_formatters);

        // initialize console logs
        init_agas_console_log(ini);
        init_parcel_console_log(ini);
        init_timing_console_log(ini);
        init_hpx_console_log(ini);
        init_app_console_log(ini);
        init_debuglog_console_log(ini);
    }
}}}    // namespace hpx::util::detail

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(pop)
#endif

#else    // HPX_HAVE_LOGGING

#include <hpx/modules/logging.hpp>
#include <hpx/runtime_configuration/runtime_configuration.hpp>
#include <hpx/util/get_entry_as.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace hpx { namespace util {
    struct console_local;
    namespace detail {
        inline void define_formatters_local() {}
        template <typename ConsoleDestination, typename DefineFormatters>
        void init_logging(runtime_configuration& ini, bool, DefineFormatters&&)
        {
            // warn if logging is requested

            if (util::get_entry_as<int>(ini, "hpx.logging.level", -1) > 0 ||
                util::get_entry_as<int>(ini, "hpx.logging.timing.level", -1) >
                    0 ||
                util::get_entry_as<int>(ini, "hpx.logging.agas.level", -1) >
                    0 ||
                util::get_entry_as<int>(ini, "hpx.logging.debuglog.level", -1) >
                    0 ||
                util::get_entry_as<int>(
                    ini, "hpx.logging.application.level", -1) > 0)
            {
                std::cerr << "hpx::init_logging: warning: logging is "
                             "requested "
                             "even "
                             "while it was disabled at compile time. If you "
                             "need logging to be functional, please "
                             "reconfigure "
                             "and "
                             "rebuild HPX with HPX_WITH_LOGGING set to ON."
                          << std::endl;
            }
        }
    }    // namespace detail
}}       // namespace hpx::util

#endif    // HPX_HAVE_LOGGING

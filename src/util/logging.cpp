//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#if !defined(HPX_NO_LOGGING)

#include <hpx/exception.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/static.hpp>
#include <hpx/runtime/components/console_logging.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>

#include <boost/version.hpp>
#include <boost/config.hpp>
#include <boost/filesystem/operations.hpp>

#include <boost/assign/std/vector.hpp>
#include <boost/logging/format/named_write.hpp>
#include <boost/logging/format/destination/defaults.hpp>

#include <cstddef>
#include <cstdlib>

#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    namespace detail
    {
        boost::logging::level::type get_log_level(std::string const& env, bool allow_always = false)
        {
            try {
                int env_val = boost::lexical_cast<int>(env);
                if (env_val < 0)
                    return boost::logging::level::disable_all;

                switch (env_val) {
                case 0:
                    return allow_always ?
                        boost::logging::level::always :
                        boost::logging::level::disable_all;
                case 1:   return boost::logging::level::fatal;
                case 2:   return boost::logging::level::error;
                case 3:   return boost::logging::level::warning;
                case 4:   return boost::logging::level::info;
                default:
                    break;
                }
                return boost::logging::level::debug;
            }
            catch (boost::bad_lexical_cast const&) {
                return boost::logging::level::disable_all;
            }
        }

        // unescape config entry
        std::string unescape(std::string const &value)
        {
            std::string result;
            std::string::size_type pos = 0;
            std::string::size_type pos1 = value.find_first_of ("\\", 0);
            if (std::string::npos != pos1) {
                do {
                    switch (value[pos1+1]) {
                    case '\\':
                    case '\"':
                    case '?':
                        result = result + value.substr(pos, pos1-pos);
                        pos1 = value.find_first_of ("\\", (pos = pos1+1)+1);
                        break;

                    case 'n':
                        result = result + value.substr(pos, pos1-pos) + "\n";
                        pos1 = value.find_first_of ("\\", pos = pos1+1);
                        ++pos;
                        break;

                    default:
                        result = result + value.substr(pos, pos1-pos+1);
                        pos1 = value.find_first_of ("\\", pos = pos1+1);
                    }

                } while (pos1 != std::string::npos);
                result = result + value.substr(pos);
            }
            else {
            // the string doesn't contain any escaped character sequences
                result = value;
            }
            return result;
        }
    }

    std::string levelname(int level)
    {
        switch (level) {
        case boost::logging::level::enable_all:
            return "     <all>";
        case boost::logging::level::debug:
            return "   <debug>";
        case boost::logging::level::info:
            return "    <info>";
        case boost::logging::level::warning:
            return " <warning>";
        case boost::logging::level::error:
            return "   <error>";
        case boost::logging::level::fatal:
            return "   <fatal>";
        case boost::logging::level::always:
            return "  <always>";
        }
        // FIXME: This one will screw up the formatting.
        //return "<" + boost::lexical_cast<std::string>(level) + ">";
        return     " <unknown>";
    }

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: shepherd
    struct shepherd_thread_id
      : boost::logging::formatter::class_<
            shepherd_thread_id,
            boost::logging::formatter::implement_op_equal::no_context>
    {
        shepherd_thread_id()
        {}

        void operator()(param str) const
        {
            std::stringstream out;
            out << std::hex << std::setw(sizeof(std::size_t)*2)
                << std::setfill('0')
                << hpx::get_worker_thread_num();
            str.prepend_string(out.str());
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: locality prefix
    struct locality_prefix
      : boost::logging::formatter::class_<
            locality_prefix,
            boost::logging::formatter::implement_op_equal::no_context>
    {
        locality_prefix()
        {}

        void operator()(param str) const
        {
            boost::uint32_t locality_id = naming::invalid_locality_id;
            applier::applier* appl = applier::get_applier_ptr();
            if (appl)
                locality_id = appl->get_locality_id();

            if (naming::invalid_locality_id != locality_id) {
                std::stringstream out;
                out << std::hex << std::setw(sizeof(boost::uint32_t)*2)
                    << std::setfill('0') << locality_id;
                str.prepend_string(out.str());
            }
            else {
                // called from outside a HPX thread
                str.prepend_string(std::string(sizeof(boost::uint32_t)*2, '-'));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: HPX thread id
    struct thread_id
      : boost::logging::formatter::class_<
            thread_id,
            boost::logging::formatter::implement_op_equal::no_context>
    {
        void operator()(param str) const
        {
            threads::thread_self* self = threads::get_self_ptr();
            if (0 != self) {
                // called from inside a HPX thread
                threads::thread_id_type id = self->get_thread_id();
                if (id != threads::invalid_thread_id) {
                    std::stringstream out;
                    out << std::hex << std::setw(sizeof(void*)*2) 
                        << std::setfill('0')
                        << reinterpret_cast<std::ptrdiff_t>(id);
                    str.prepend_string(out.str());
                    return;
                }
            }

            // called from outside a HPX thread or invalid thread id
            str.prepend_string(std::string(sizeof(void*)*2, '-'));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: HPX thread phase
    struct thread_phase
      : boost::logging::formatter::class_<
            thread_phase,
            boost::logging::formatter::implement_op_equal::no_context>
    {
        void operator()(param str) const
        {
            threads::thread_self* self = threads::get_self_ptr();
            if (0 != self) {
                // called from inside a HPX thread
                std::size_t phase = self->get_thread_phase();
                if (0 != phase) {
                    std::stringstream out;
                    out << std::hex << std::setw(2) << std::setfill('0')
                        << self->get_thread_phase();
                    str.prepend_string(out.str());
                    return;
                }
            }

            // called from outside a HPX thread or no phase given
            str.prepend_string("--");
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: locality prefix of parent thread
    struct parent_thread_locality
      : boost::logging::formatter::class_<
            parent_thread_locality,
            boost::logging::formatter::implement_op_equal::no_context>
    {
        void operator()(param str) const
        {
            boost::uint32_t parent_locality_id = threads::get_parent_locality_id();
            if (naming::invalid_locality_id != parent_locality_id) {
                // called from inside a HPX thread
                std::stringstream out;
                out << std::hex << std::setw(sizeof(boost::uint32_t)*2) << std::setfill('0')
                    << parent_locality_id;
                str.prepend_string(out.str());
            }
            else {
                // called from outside a HPX thread
                str.prepend_string(std::string(sizeof(boost::uint32_t)*2, '-'));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: HPX parent thread id
    struct parent_thread_id
      : boost::logging::formatter::class_<
            parent_thread_id,
            boost::logging::formatter::implement_op_equal::no_context>
    {
        void operator()(param str) const
        {
            threads::thread_id_type parent_id = threads::get_parent_id();
            if (0 != parent_id && threads::invalid_thread_id != parent_id) {
                // called from inside a HPX thread
                std::stringstream out;
                out << std::hex << std::setw(sizeof(void*)*2) 
                    << std::setfill('0')
                    << reinterpret_cast<std::ptrdiff_t>(parent_id);
                str.prepend_string(out.str());
            }
            else {
                // called from outside a HPX thread
                str.prepend_string(std::string(sizeof(void*)*2, '-'));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: HPX parent thread phase
    struct parent_thread_phase
      : boost::logging::formatter::class_<
            parent_thread_phase,
            boost::logging::formatter::implement_op_equal::no_context>
    {
        void operator()(param str) const
        {
            std::size_t parent_phase = threads::get_parent_phase();
            if (0 != parent_phase) {
                // called from inside a HPX thread
                std::stringstream out;
                out << std::hex << std::setw(2) << std::setfill('0')
                    << parent_phase;
                str.prepend_string(out.str());
            }
            else {
                // called from outside a HPX thread
                str.prepend_string("--");
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: HPX component id of current thread
    struct thread_component_id
      : boost::logging::formatter::class_<
            thread_component_id,
            boost::logging::formatter::implement_op_equal::no_context>
    {
        void operator()(param str) const
        {
            boost::uint64_t component_id = threads::get_self_component_id();
            if (0 != component_id) {
                // called from inside a HPX thread
                std::stringstream out;
                out << std::hex << std::setw(sizeof(boost::uint64_t)*2) << std::setfill('0')
                    << component_id;
                str.prepend_string(out.str());
            }
            else {
                // called from outside a HPX thread
                str.prepend_string(std::string(sizeof(boost::uint64_t)*2, '-'));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom log destination: send generated strings to console
    struct console : boost::logging::destination::is_generic
    {
        console(std::size_t level, logging_destination dest)
          : level_(level), dest_(dest)
        {}

        template<typename MsgType>
        void operator()(MsgType const& msg) const
        {
            components::console_logging(dest_, level_, msg);
        }

        bool operator==(console const& rhs) const
        {
            return dest_ == rhs.dest_;
        }

        std::size_t level_;
        logging_destination dest_;
    };

#if defined(ANDROID) || defined(__ANDROID__)
    // default log destination for Android
    struct android_log : boost::logging::destination::is_generic
    {
        android_log(char const* tag_)
          : tag(tag_)
        {}

        template<typename MsgType>
        void operator()(MsgType const& msg) const
        {
            __android_log_write(ANDROID_LOG_DEBUG, tag.c_str(), msg.c_str());
        }

        bool operator==(console const& rhs) const
        {
            return tag == rhs.tag;
        }

        std::string tag;
    };
#endif

    ///////////////////////////////////////////////////////////////////////////
    // this is required in order to use the logging library
    BOOST_DEFINE_LOG_FILTER_WITH_ARGS(agas_level, filter_type,
        boost::logging::level::disable_all)
    BOOST_DEFINE_LOG(agas_logger, logger_type)

    // initialize logging for AGAS
    void init_agas_log(util::section const& ini, bool isconsole)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.agas")) {
            util::section const* logini = ini.get_section("hpx.logging.agas");
            BOOST_ASSERT(NULL != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            logdest = logini->get_entry("destination", empty);
            logformat = detail::unescape(logini->get_entry("format", empty));
        }

        unsigned lvl = boost::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel);

        if (boost::logging::level::disable_all != lvl)
        {
#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())      // ensure minimal defaults
                logdest = isconsole ? "android_log" : "console";
            agas_logger()->writer().add_destination("android_log", 
                android_log("hpx.agas"));
#else
            if (logdest.empty())      // ensure minimal defaults
                logdest = isconsole ? "cerr" : "console";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            agas_logger()->writer().add_destination("console",
                console(lvl, destination_agas));
            agas_logger()->writer().write(logformat, logdest);
            agas_logger()->writer().replace_formatter("osthread", shepherd_thread_id());
            agas_logger()->writer().replace_formatter("locality", locality_prefix());
            agas_logger()->writer().replace_formatter("hpxthread", thread_id());
            agas_logger()->writer().replace_formatter("hpxphase", thread_phase());
            agas_logger()->writer().replace_formatter("hpxparent", parent_thread_id());
            agas_logger()->writer().replace_formatter("hpxparentphase", parent_thread_phase());
            agas_logger()->writer().replace_formatter("parentloc", parent_thread_locality());
            agas_logger()->writer().replace_formatter("hpxcomponent", thread_component_id());
            agas_logger()->mark_as_initialized();
            agas_level()->set_enabled(lvl);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // this is required in order to use the logging library for timings
    BOOST_DEFINE_LOG_FILTER_WITH_ARGS(timing_level, filter_type,
        boost::logging::level::disable_all)
    BOOST_DEFINE_LOG(timing_logger, logger_type)

    // initialize logging for performance measurements
    void init_timing_log(util::section const& ini, bool isconsole)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.timing")) {
            util::section const* logini = ini.get_section("hpx.logging.timing");
            BOOST_ASSERT(NULL != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            logdest = logini->get_entry("destination", empty);
            logformat = detail::unescape(logini->get_entry("format", empty));
        }

        unsigned lvl = boost::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel);

        if (boost::logging::level::disable_all != lvl)
        {
#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())      // ensure minimal defaults
                logdest = isconsole ? "android_log" : "console";
            timing_logger()->writer().add_destination("android_log", 
                android_log("hpx.timing"));
#else
            if (logdest.empty())      // ensure minimal defaults
                logdest = isconsole ? "cerr" : "console";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            timing_logger()->writer().add_destination("console",
                console(lvl, destination_timing));
            timing_logger()->writer().write(logformat, logdest);
            timing_logger()->writer().replace_formatter("osthread", shepherd_thread_id());
            timing_logger()->writer().replace_formatter("locality", locality_prefix());
            timing_logger()->writer().replace_formatter("hpxthread", thread_id());
            timing_logger()->writer().replace_formatter("hpxphase", thread_phase());
            timing_logger()->writer().replace_formatter("hpxparent", parent_thread_id());
            timing_logger()->writer().replace_formatter("hpxparentphase", parent_thread_phase());
            timing_logger()->writer().replace_formatter("parentloc", parent_thread_locality());
            timing_logger()->writer().replace_formatter("hpxcomponent", thread_component_id());
            timing_logger()->mark_as_initialized();
            timing_level()->set_enabled(lvl);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    BOOST_DEFINE_LOG_FILTER_WITH_ARGS(hpx_level, filter_type,
        boost::logging::level::disable_all)
    BOOST_DEFINE_LOG(hpx_logger, logger_type)

    BOOST_DEFINE_LOG_FILTER_WITH_ARGS(hpx_error_level, filter_type,
        boost::logging::level::fatal)
    BOOST_DEFINE_LOG(hpx_error_logger, logger_type)

    void init_hpx_logs(util::section const& ini, bool isconsole)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging")) {
            util::section const* logini = ini.get_section("hpx.logging");
            BOOST_ASSERT(NULL != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            logdest = logini->get_entry("destination", empty);
            logformat = detail::unescape(logini->get_entry("format", empty));
        }

        unsigned lvl = boost::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel, true);

#if defined(ANDROID) || defined(__ANDROID__)
        if (logdest.empty())      // ensure minimal defaults
            logdest = isconsole ? "android_log" : "console";
        hpx_logger()->writer().add_destination("android_log", android_log("hpx"));
        hpx_error_logger()->writer().add_destination("android_log", 
            android_log("hpx"));
#else
        if (logdest.empty())      // ensure minimal defaults
            logdest = isconsole ? "cerr" : "console";
#endif
        if (logformat.empty())
            logformat = "|\\n";

        if (boost::logging::level::disable_all != lvl) {
            hpx_logger()->writer().add_destination("console",
                console(lvl, destination_hpx));
            hpx_logger()->writer().write(logformat, logdest);
            hpx_logger()->writer().replace_formatter("osthread", shepherd_thread_id());
            hpx_logger()->writer().replace_formatter("locality", locality_prefix());
            hpx_logger()->writer().replace_formatter("hpxthread", thread_id());
            hpx_logger()->writer().replace_formatter("hpxphase", thread_phase());
            hpx_logger()->writer().replace_formatter("hpxparent", parent_thread_id());
            hpx_logger()->writer().replace_formatter("hpxparentphase", parent_thread_phase());
            hpx_logger()->writer().replace_formatter("parentloc", parent_thread_locality());
            hpx_logger()->writer().replace_formatter("hpxcomponent", thread_component_id());
            hpx_logger()->mark_as_initialized();
            hpx_level()->set_enabled(lvl);

            // errors are logged to the given destination and to cerr
            hpx_error_logger()->writer().add_destination("console",
                console(lvl, destination_hpx));
#if !defined(ANDROID) && !defined(__ANDROID__)
            if (logdest != "cerr")
                hpx_error_logger()->writer().write(logformat, logdest + " cerr");
#endif
            hpx_error_logger()->writer().replace_formatter("osthread", shepherd_thread_id());
            hpx_error_logger()->writer().replace_formatter("locality", locality_prefix());
            hpx_error_logger()->writer().replace_formatter("hpxthread", thread_id());
            hpx_error_logger()->writer().replace_formatter("hpxphase", thread_phase());
            hpx_error_logger()->writer().replace_formatter("hpxparent", parent_thread_id());
            hpx_error_logger()->writer().replace_formatter("hpxparentphase", parent_thread_phase());
            hpx_error_logger()->writer().replace_formatter("parentloc", parent_thread_locality());
            hpx_error_logger()->writer().replace_formatter("hpxcomponent", thread_component_id());
            hpx_error_logger()->mark_as_initialized();
            hpx_error_level()->set_enabled(lvl);
        }
        else {
            // errors are always logged to cerr
            if (!isconsole) {
                hpx_error_logger()->writer().add_destination("console",
                    console(lvl, destination_hpx));
                hpx_error_logger()->writer().write(logformat, "console");
            }
            else {
#if defined(ANDROID) || defined(__ANDROID__)
                hpx_error_logger()->writer().write(logformat, "android_log");
#else
                hpx_error_logger()->writer().write(logformat, "cerr");
#endif
            }
            hpx_error_logger()->writer().replace_formatter("osthread", shepherd_thread_id());
            hpx_error_logger()->writer().replace_formatter("locality", locality_prefix());
            hpx_error_logger()->writer().replace_formatter("hpxthread", thread_id());
            hpx_error_logger()->writer().replace_formatter("hpxphase", thread_phase());
            hpx_error_logger()->writer().replace_formatter("hpxparent", parent_thread_id());
            hpx_error_logger()->writer().replace_formatter("hpxparentphase", parent_thread_phase());
            hpx_error_logger()->writer().replace_formatter("parentloc", parent_thread_locality());
            hpx_error_logger()->writer().replace_formatter("hpxcomponent", thread_component_id());
            hpx_error_logger()->mark_as_initialized();
            hpx_error_level()->set_enabled(boost::logging::level::fatal);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    BOOST_DEFINE_LOG_FILTER_WITH_ARGS(app_level, filter_type,
        boost::logging::level::disable_all)
    BOOST_DEFINE_LOG(app_logger, logger_type)

    BOOST_DEFINE_LOG_FILTER_WITH_ARGS(app_error_level, filter_type,
        boost::logging::level::fatal)
    BOOST_DEFINE_LOG(app_error_logger, logger_type)

    // initialize logging for application
    void init_app_logs(util::section const& ini, bool isconsole)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.application")) {
            util::section const* logini = ini.get_section("hpx.logging.application");
            BOOST_ASSERT(NULL != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            logdest = logini->get_entry("destination", empty);
            logformat = detail::unescape(logini->get_entry("format", empty));
        }

        unsigned lvl = boost::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel);

        if (boost::logging::level::disable_all != lvl) {
#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())      // ensure minimal defaults
                logdest = isconsole ? "android_log" : "console";
            app_logger()->writer().add_destination("android_log", 
                android_log("hpx.application"));
#else
            if (logdest.empty())      // ensure minimal defaults
                logdest = isconsole ? "cerr" : "console";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            app_logger()->writer().add_destination("console",
                console(lvl, destination_app));
            app_logger()->writer().write(logformat, logdest);
            app_logger()->writer().replace_formatter("osthread", shepherd_thread_id());
            app_logger()->writer().replace_formatter("locality", locality_prefix());
            app_logger()->writer().replace_formatter("hpxthread", thread_id());
            app_logger()->writer().replace_formatter("hpxphase", thread_phase());
            app_logger()->writer().replace_formatter("hpxparent", parent_thread_id());
            app_logger()->writer().replace_formatter("hpxparentphase", parent_thread_phase());
            app_logger()->writer().replace_formatter("parentloc", parent_thread_locality());
            app_logger()->writer().replace_formatter("hpxcomponent", thread_component_id());
            app_logger()->mark_as_initialized();
            app_level()->set_enabled(lvl);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    BOOST_DEFINE_LOG_FILTER_WITH_ARGS(agas_console_level, filter_type,
        boost::logging::level::disable_all)
    BOOST_DEFINE_LOG(agas_console_logger, logger_type)

    // initialize logging for AGAS
    void init_agas_console_log(util::section const& ini)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.console.agas")) {
            util::section const* logini = ini.get_section("hpx.logging.console.agas");
            BOOST_ASSERT(NULL != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            logdest = logini->get_entry("destination", empty);
            logformat = detail::unescape(logini->get_entry("format", empty));
        }

        unsigned lvl = boost::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel, true);

        if (boost::logging::level::disable_all != lvl)
        {
#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())      // ensure minimal defaults
                logdest = "android_log";
            agas_console_logger()->writer().add_destination("android_log", 
                android_log("hpx.agas"));
#else
            if (logdest.empty())      // ensure minimal defaults
                logdest = "cerr";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            agas_console_logger()->writer().write(logformat, logdest);
            agas_console_logger()->mark_as_initialized();
            agas_console_level()->set_enabled(lvl);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    BOOST_DEFINE_LOG_FILTER_WITH_ARGS(timing_console_level, filter_type,
        boost::logging::level::disable_all)
    BOOST_DEFINE_LOG(timing_console_logger, logger_type)

    // initialize logging for performance measurements
    void init_timing_console_log(util::section const& ini)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.console.timing")) {
            util::section const* logini = ini.get_section("hpx.logging.console.timing");
            BOOST_ASSERT(NULL != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            logdest = logini->get_entry("destination", empty);
            logformat = detail::unescape(logini->get_entry("format", empty));
        }

        unsigned lvl = boost::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel, true);

        if (boost::logging::level::disable_all != lvl)
        {
#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())      // ensure minimal defaults
                logdest = "android_log";
            timing_console_logger()->writer().add_destination("android_log", 
                android_log("hpx.timing"));
#else
            if (logdest.empty())      // ensure minimal defaults
                logdest = "cerr";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            timing_console_logger()->writer().write(logformat, logdest);
            timing_console_logger()->mark_as_initialized();
            timing_console_level()->set_enabled(lvl);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    BOOST_DEFINE_LOG_FILTER_WITH_ARGS(hpx_console_level, filter_type,
        boost::logging::level::disable_all)
    BOOST_DEFINE_LOG(hpx_console_logger, logger_type)

    // initialize logging for HPX runtime
    void init_hpx_console_log(util::section const& ini)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.console")) {
            util::section const* logini = ini.get_section("hpx.logging.console");
            BOOST_ASSERT(NULL != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            logdest = logini->get_entry("destination", empty);
            logformat = detail::unescape(logini->get_entry("format", empty));
        }

        unsigned lvl = boost::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel, true);

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())      // ensure minimal defaults
                logdest = "android_log";
            hpx_console_logger()->writer().add_destination("android_log", 
                android_log("hpx"));
#else
            if (logdest.empty())      // ensure minimal defaults
                logdest = "cerr";
#endif
        if (logformat.empty())
            logformat = "|";

        if (boost::logging::level::disable_all != lvl) {
            hpx_console_logger()->writer().write(logformat, logdest);
            hpx_console_logger()->mark_as_initialized();
            hpx_console_level()->set_enabled(lvl);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    BOOST_DEFINE_LOG_FILTER_WITH_ARGS(app_console_level, filter_type,
        boost::logging::level::disable_all)
    BOOST_DEFINE_LOG(app_console_logger, logger_type)

    // initialize logging for applications
    void init_app_console_log(util::section const& ini)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.console.application")) {
            util::section const* logini = ini.get_section("hpx.logging.console.application");
            BOOST_ASSERT(NULL != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            logdest = logini->get_entry("destination", empty);
            logformat = detail::unescape(logini->get_entry("format", empty));
        }

        unsigned lvl = boost::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel, true);

        if (boost::logging::level::disable_all != lvl)
        {
#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())      // ensure minimal defaults
                logdest = "android_log";
            app_console_logger()->writer().add_destination("android_log", 
                android_log("hpx.application"));
#else
            if (logdest.empty())      // ensure minimal defaults
                logdest = "cerr";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            app_console_logger()->writer().write(logformat, logdest);
            app_console_logger()->mark_as_initialized();
            app_console_level()->set_enabled(lvl);
        }
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // the logging_configuration type will be instantiated exactly once
    struct logging_configuration
    {
        logging_configuration();
        std::vector<std::string> prefill_;
    };

// define the format for the generated time stamps
#define HPX_TIMEFORMAT "$hh:$mm.$ss.$mili"

    logging_configuration::logging_configuration()
    {
        try {
            // add default logging configuration as defaults to the ini data
            // this will be overwritten by related entries in the read hpx.ini
            using namespace boost::assign;
            prefill_ +=
                // general logging
                "[hpx.logging]",
                "level = ${HPX_LOGLEVEL:0}",
                "destination = ${HPX_LOGDESTINATION:console}",
                "format = ${HPX_LOGFORMAT:"
                    "(T%locality%/%hpxthread%.%hpxphase%/%hpxcomponent%) "
                    "P%parentloc%/%hpxparent%.%hpxparentphase% %time%(" HPX_TIMEFORMAT
                    ") [%idx%]|\\n}",

                // general console logging
                "[hpx.logging.console]",
                "level = ${HPX_LOGLEVEL:$[hpx.logging.level]}",
#if defined(ANDROID) || defined(__ANDROID__)
                "destination = ${HPX_CONSOLE_LOGDESTINATION:android_log}",
#else
                "destination = ${HPX_CONSOLE_LOGDESTINATION:file(hpx.$[system.pid].log)}",
#endif
                "format = ${HPX_CONSOLE_LOGFORMAT:|}",

                // logging related to timing
                "[hpx.logging.timing]",
                "level = ${HPX_TIMING_LOGLEVEL:-1}",
                "destination = ${HPX_TIMING_LOGDESTINATION:console}",
                "format = ${HPX_TIMING_LOGFORMAT:"
                    "(T%locality%/%hpxthread%.%hpxphase%/%hpxcomponent%) "
                    "P%parentloc%/%hpxparent%.%hpxparentphase% %time%(" HPX_TIMEFORMAT
                    ") [%idx%] [TIM] |\\n}",

                // console logging related to timing
                "[hpx.logging.console.timing]",
                "level = ${HPX_TIMING_LOGLEVEL:$[hpx.logging.timing.level]}",
#if defined(ANDROID) || defined(__ANDROID__)
                "destination = ${HPX_CONSOLE_TIMING_LOGDESTINATION:android_log}",
#else
                "destination = ${HPX_CONSOLE_TIMING_LOGDESTINATION:file(hpx.timing.$[system.pid].log)}",
#endif
                "format = ${HPX_CONSOLE_TIMING_LOGFORMAT:|}",

                // logging related to AGAS
                "[hpx.logging.agas]",
                "level = ${HPX_AGAS_LOGLEVEL:-1}",
//                     "destination = ${HPX_AGAS_LOGDESTINATION:console}",
                "destination = ${HPX_AGAS_LOGDESTINATION:file(hpx.agas.$[system.pid].log)}",
                "format = ${HPX_AGAS_LOGFORMAT:"
                    "(T%locality%/%hpxthread%.%hpxphase%/%hpxcomponent%) "
                    "P%parentloc%/%hpxparent%.%hpxparentphase% %time%(" HPX_TIMEFORMAT
                    ") [%idx%][AGAS] |\\n}",

                // console logging related to AGAS
                "[hpx.logging.console.agas]",
                "level = ${HPX_AGAS_LOGLEVEL:$[hpx.logging.agas.level]}",
#if defined(ANDROID) || defined(__ANDROID__)
                "destination = ${HPX_CONSOLE_AGAS_LOGDESTINATION:android_log}",
#else
                "destination = ${HPX_CONSOLE_AGAS_LOGDESTINATION:file(hpx.agas.$[system.pid].log)}",
#endif
                "format = ${HPX_CONSOLE_AGAS_LOGFORMAT:|}",

                // logging related to applications
                "[hpx.logging.application]",
                "level = ${HPX_APP_LOGLEVEL:-1}",
                "destination = ${HPX_APP_LOGDESTINATION:console}",
                "format = ${HPX_APP_LOGFORMAT:"
                    "(T%locality%/%hpxthread%.%hpxphase%/%hpxcomponent%) "
                    "P%parentloc%/%hpxparent%.%hpxparentphase% %time%(" HPX_TIMEFORMAT
                    ") [%idx%] [APP] |\\n}",

                // console logging related to applications
                "[hpx.logging.console.application]",
                "level = ${HPX_APP_LOGLEVEL:$[hpx.logging.application.level]}",
#if defined(ANDROID) || defined(__ANDROID__)
                "destination = ${HPX_CONSOLE_APP_LOGDESTINATION:android_log}",
#else
                "destination = ${HPX_CONSOLE_APP_LOGDESTINATION:file(hpx.application.$[system.pid].log)}",
#endif
                "format = ${HPX_CONSOLE_APP_LOGFORMAT:|}"
            ;
        }
        catch (std::exception const&) {
            // just in case something goes wrong
            std::cerr << "caught std::exception during initialization"
                      << std::endl;
        }
    }

#undef HPX_TIMEFORMAT

    struct init_logging_tag {};
    std::vector<std::string> const& get_logging_data()
    {
        static_<logging_configuration, init_logging_tag> init;
        return init.get().prefill_;
    }

    ///////////////////////////////////////////////////////////////////////////
    init_logging::init_logging(runtime_configuration& ini, bool isconsole,
        naming::resolver_client& agas_client)
    {
        init_agas_log(ini, isconsole);
        init_timing_log(ini, isconsole);
        init_hpx_logs(ini, isconsole);
        init_app_logs(ini, isconsole);

        // initialize console logs
        init_agas_console_log(ini);
        init_timing_console_log(ini);
        init_hpx_console_log(ini);
        init_app_console_log(ini);
    }
}}}

#else  // HPX_NO_LOGGING

#include <hpx/util/logging.hpp>

namespace hpx { namespace util { namespace detail
{
    dummy_log_impl dummy_log = dummy_log_impl();

    std::vector<std::string> get_logging_data()
    {
        static std::vector<std::string> dummy_data;
        return dummy_data;
    }
}}}

#endif // HPX_NO_LOGGING

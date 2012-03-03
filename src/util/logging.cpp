//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstddef>
#include <cstdlib>

#include <hpx/hpx_fwd.hpp>
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

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    namespace detail
    {
        int get_log_level(std::string const& env, bool allow_always = false)
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
            boost::logging::formatter::implement_op_equal::no_context
        >
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
            boost::logging::formatter::implement_op_equal::no_context
        >
    {
        locality_prefix()
        {}

        void operator()(param str) const
        {
            boost::uint32_t locality_id = 0;
            applier::applier* appl = applier::get_applier_ptr();
            if (appl)
                locality_id = appl->get_locality_id();

            std::stringstream out;
            out << std::hex << std::setw(sizeof(boost::uint32_t)*2)
                << std::setfill('0') << locality_id;
            str.prepend_string(out.str());
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: PX thread id
    struct thread_id
      : boost::logging::formatter::class_<
            thread_id,
            boost::logging::formatter::implement_op_equal::no_context>
    {
        void operator()(param str) const
        {
            threads::thread_self* self = threads::get_self_ptr();
            if (0 != self) {
                // called from inside a PX thread
                std::stringstream out;
                out << std::hex << std::setw(sizeof(void*)*2) << std::setfill('0')
                    << reinterpret_cast<std::ptrdiff_t>(self->get_thread_id());
                str.prepend_string(out.str());
            }
            else {
                // called from outside a PX thread
                str.prepend_string(std::string(sizeof(void*)*2, '-'));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: PX thread phase
    struct thread_phase
      : boost::logging::formatter::class_<
            thread_phase,
            boost::logging::formatter::implement_op_equal::no_context>
    {
        void operator()(param str) const
        {
            threads::thread_self* self = threads::get_self_ptr();
            if (0 != self) {
                // called from inside a PX thread
                std::stringstream out;
                out << std::hex << std::setw(2) << std::setfill('0')
                    << self->get_thread_phase();
                str.prepend_string(out.str());
            }
            else {
                // called from outside a PX thread
                str.prepend_string("--");
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: locality prefix of parent thread
    struct parent_thread_locality
      : boost::logging::formatter::class_<
            parent_thread_locality,
            boost::logging::formatter::implement_op_equal::no_context
        >
    {
        void operator()(param str) const
        {
            boost::uint32_t parent_prefix = threads::get_parent_prefix();
            if (0 != parent_prefix) {
                // called from inside a PX thread
                std::stringstream out;
                out << std::hex << std::setw(sizeof(boost::uint32_t)*2) << std::setfill('0')
                    << parent_prefix;
                str.prepend_string(out.str());
            }
            else {
                // called from outside a PX thread
                str.prepend_string(std::string(sizeof(boost::uint32_t)*2, '-'));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: PX parent thread id
    struct parent_thread_id
      : boost::logging::formatter::class_<
            parent_thread_id,
            boost::logging::formatter::implement_op_equal::no_context
        >
    {
        void operator()(param str) const
        {
            threads::thread_id_type parent_id = threads::get_parent_id();
            if (0 != parent_id) {
                // called from inside a PX thread
                std::stringstream out;
                out << std::hex << std::setw(sizeof(void*)*2) << std::setfill('0')
                    << reinterpret_cast<std::ptrdiff_t>(parent_id);
                str.prepend_string(out.str());
            }
            else {
                // called from outside a PX thread
                str.prepend_string(std::string(sizeof(void*)*2, '-'));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: PX parent thread phase
    struct parent_thread_phase
      : boost::logging::formatter::class_<
            parent_thread_phase,
            boost::logging::formatter::implement_op_equal::no_context
        >
    {
        void operator()(param str) const
        {
            std::size_t parent_phase = threads::get_parent_phase();
            if (0 != parent_phase) {
                // called from inside a PX thread
                std::stringstream out;
                out << std::hex << std::setw(2) << std::setfill('0')
                    << parent_phase;
                str.prepend_string(out.str());
            }
            else {
                // called from outside a PX thread
                str.prepend_string("--");
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: PX component id of current thread
    struct thread_component_id
      : boost::logging::formatter::class_<
            thread_component_id,
            boost::logging::formatter::implement_op_equal::no_context
        >
    {
        void operator()(param str) const
        {
            boost::uint64_t component_id = threads::get_self_component_id();
            if (0 != component_id) {
                // called from inside a PX thread
                std::stringstream out;
                out << std::hex << std::setw(sizeof(boost::uint64_t)*2) << std::setfill('0')
                    << component_id;
                str.prepend_string(out.str());
            }
            else {
                // called from outside a PX thread
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
            if (logdest.empty())      // ensure minimal defaults
                logdest = isconsole ? "cerr" : "console";
            if (logformat.empty())
                logformat = "|\\n";

            agas_logger()->writer().add_destination("console",
                console(lvl, destination_agas));
            agas_logger()->writer().write(logformat, logdest);
            agas_logger()->writer().replace_formatter("shepherd", shepherd_thread_id());
            agas_logger()->writer().replace_formatter("locality", locality_prefix());
            agas_logger()->writer().replace_formatter("pxthread", thread_id());
            agas_logger()->writer().replace_formatter("pxphase", thread_phase());
            agas_logger()->writer().replace_formatter("pxparent", parent_thread_id());
            agas_logger()->writer().replace_formatter("pxparentphase", parent_thread_phase());
            agas_logger()->writer().replace_formatter("parentloc", parent_thread_locality());
            agas_logger()->writer().replace_formatter("pxcomponent", thread_component_id());
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
            if (logdest.empty())      // ensure minimal defaults
                logdest = isconsole ? "cerr" : "console";
            if (logformat.empty())
                logformat = "|\\n";

            timing_logger()->writer().add_destination("console",
                console(lvl, destination_timing));
            timing_logger()->writer().write(logformat, logdest);
            timing_logger()->writer().replace_formatter("shepherd", shepherd_thread_id());
            timing_logger()->writer().replace_formatter("locality", locality_prefix());
            timing_logger()->writer().replace_formatter("pxthread", thread_id());
            timing_logger()->writer().replace_formatter("pxphase", thread_phase());
            timing_logger()->writer().replace_formatter("pxparent", parent_thread_id());
            timing_logger()->writer().replace_formatter("pxparentphase", parent_thread_phase());
            timing_logger()->writer().replace_formatter("parentloc", parent_thread_locality());
            timing_logger()->writer().replace_formatter("pxcomponent", thread_component_id());
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

        if (logdest.empty())      // ensure minimal defaults
            logdest = isconsole ? "cerr" : "console";
        if (logformat.empty())
            logformat = "|\\n";

        if (boost::logging::level::disable_all != lvl) {
            hpx_logger()->writer().add_destination("console",
                console(lvl, destination_hpx));
            hpx_logger()->writer().write(logformat, logdest);
            hpx_logger()->writer().replace_formatter("shepherd", shepherd_thread_id());
            hpx_logger()->writer().replace_formatter("locality", locality_prefix());
            hpx_logger()->writer().replace_formatter("pxthread", thread_id());
            hpx_logger()->writer().replace_formatter("pxphase", thread_phase());
            hpx_logger()->writer().replace_formatter("pxparent", parent_thread_id());
            hpx_logger()->writer().replace_formatter("pxparentphase", parent_thread_phase());
            hpx_logger()->writer().replace_formatter("parentloc", parent_thread_locality());
            hpx_logger()->writer().replace_formatter("pxcomponent", thread_component_id());
            hpx_logger()->mark_as_initialized();
            hpx_level()->set_enabled(lvl);

            // errors are logged to the given destination and to cerr
            hpx_error_logger()->writer().add_destination("console",
                console(lvl, destination_hpx));
            hpx_error_logger()->writer().write(logformat, logdest + " cerr");
            hpx_error_logger()->writer().replace_formatter("shepherd", shepherd_thread_id());
            hpx_error_logger()->writer().replace_formatter("locality", locality_prefix());
            hpx_error_logger()->writer().replace_formatter("pxthread", thread_id());
            hpx_error_logger()->writer().replace_formatter("pxphase", thread_phase());
            hpx_error_logger()->writer().replace_formatter("pxparent", parent_thread_id());
            hpx_error_logger()->writer().replace_formatter("pxparentphase", parent_thread_phase());
            hpx_error_logger()->writer().replace_formatter("parentloc", parent_thread_locality());
            hpx_error_logger()->writer().replace_formatter("pxcomponent", thread_component_id());
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
                hpx_error_logger()->writer().write(logformat, "cerr");
            }
            hpx_error_logger()->writer().replace_formatter("shepherd", shepherd_thread_id());
            hpx_error_logger()->writer().replace_formatter("locality", locality_prefix());
            hpx_error_logger()->writer().replace_formatter("pxthread", thread_id());
            hpx_error_logger()->writer().replace_formatter("pxphase", thread_phase());
            hpx_error_logger()->writer().replace_formatter("pxparent", parent_thread_id());
            hpx_error_logger()->writer().replace_formatter("pxparentphase", parent_thread_phase());
            hpx_error_logger()->writer().replace_formatter("parentloc", parent_thread_locality());
            hpx_error_logger()->writer().replace_formatter("pxcomponent", thread_component_id());
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
            if (logdest.empty())      // ensure minimal defaults
                logdest = isconsole ? "cerr" : "console";
            if (logformat.empty())
                logformat = "|\\n";

            app_logger()->writer().add_destination("console",
                console(lvl, destination_app));
            app_logger()->writer().write(logformat, logdest);
            app_logger()->writer().replace_formatter("shepherd", shepherd_thread_id());
            app_logger()->writer().replace_formatter("locality", locality_prefix());
            app_logger()->writer().replace_formatter("pxthread", thread_id());
            app_logger()->writer().replace_formatter("pxphase", thread_phase());
            app_logger()->writer().replace_formatter("pxparent", parent_thread_id());
            app_logger()->writer().replace_formatter("pxparentphase", parent_thread_phase());
            app_logger()->writer().replace_formatter("parentloc", parent_thread_locality());
            app_logger()->writer().replace_formatter("pxcomponent", thread_component_id());
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
            if (logdest.empty())      // ensure minimal defaults
                logdest = "cerr";
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
            if (logdest.empty())      // ensure minimal defaults
                logdest = "cerr";
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

        if (logdest.empty())      // ensure minimal defaults
            logdest = "cerr";
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
            if (logdest.empty())      // ensure minimal defaults
                logdest = "cerr";
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

// the Boost.Log generates bogus milliseconds on Linux systems
#if defined(BOOST_WINDOWS)
#define HPX_TIMEFORMAT "$hh:$mm.$ss.$mili"
#else
#define HPX_TIMEFORMAT "$hh:$mm.$ss.$mili"
#endif

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
                    "(T%locality%/%pxthread%.%pxphase%/%pxcomponent%) "
                    "P%parentloc%/%pxparent%.%pxparentphase% %time%(" HPX_TIMEFORMAT
                    ") [%idx%]|\\n}",

                // general console logging
                "[hpx.logging.console]",
                "level = ${HPX_LOGLEVEL:$[hpx.logging.level]}",
                "destination = ${HPX_CONSOLE_LOGDESTINATION:file(hpx.$[system.pid].log)}",
                "format = ${HPX_CONSOLE_LOGFORMAT:|}",

                // logging related to timing
                "[hpx.logging.timing]",
                "level = ${HPX_TIMING_LOGLEVEL:-1}",
                "destination = ${HPX_TIMING_LOGDESTINATION:console}",
                "format = ${HPX_TIMING_LOGFORMAT:"
                    "(T%locality%/%pxthread%.%pxphase%/%pxcomponent%) "
                    "P%parentloc%/%pxparent%.%pxparentphase% %time%(" HPX_TIMEFORMAT
                    ") [%idx%] [TIM] |\\n}",

                // console logging related to timing
                "[hpx.logging.console.timing]",
                "level = ${HPX_TIMING_LOGLEVEL:$[hpx.logging.timing.level]}",
                "destination = ${HPX_CONSOLE_TIMING_LOGDESTINATION:file(hpx.timing.$[system.pid].log)}",
                "format = ${HPX_CONSOLE_TIMING_LOGFORMAT:|}",

                // logging related to AGAS
                "[hpx.logging.agas]",
                "level = ${HPX_AGAS_LOGLEVEL:-1}",
//                     "destination = ${HPX_AGAS_LOGDESTINATION:console}",
                "destination = ${HPX_AGAS_LOGDESTINATION:file(hpx.agas.$[system.pid].log)}",
                "format = ${HPX_AGAS_LOGFORMAT:"
                    "(T%locality%/%pxthread%.%pxphase%/%pxcomponent%) "
                    "P%parentloc%/%pxparent%.%pxparentphase% %time%(" HPX_TIMEFORMAT
                    ") [%idx%][AGAS] |\\n}",

                // console logging related to AGAS
                "[hpx.logging.console.agas]",
                "level = ${HPX_AGAS_LOGLEVEL:$[hpx.logging.agas.level]}",
                "destination = ${HPX_CONSOLE_AGAS_LOGDESTINATION:file(hpx.agas.$[system.pid].log)}",
                "format = ${HPX_CONSOLE_AGAS_LOGFORMAT:|}",

                // logging related to applications
                "[hpx.logging.application]",
                "level = ${HPX_APP_LOGLEVEL:-1}",
                "destination = ${HPX_APP_LOGDESTINATION:console}",
                "format = ${HPX_APP_LOGFORMAT:"
                    "(T%locality%/%pxthread%.%pxphase%/%pxcomponent%) "
                    "P%parentloc%/%pxparent%.%pxparentphase% %time%(" HPX_TIMEFORMAT
                    ") [%idx%] [APP] |\\n}",

                // console logging related to applications
                "[hpx.logging.console.application]",
                "level = ${HPX_APP_LOGLEVEL:$[hpx.logging.application.level]}",
                "destination = ${HPX_CONSOLE_APP_LOGDESTINATION:file(hpx.application.$[system.pid].log)}",
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

///////////////////////////////////////////////////////////////////////////////
}}}


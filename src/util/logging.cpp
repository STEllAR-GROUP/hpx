//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstdlib>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/static.hpp>
#include <hpx/runtime/components/console_logging.hpp>

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
        int get_log_level(std::string const& env)
        {
            try {
                int env_val = boost::lexical_cast<int>(env);
                if (env_val <= 0)
                    return boost::logging::level::disable_all;

                switch (env_val) {
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
        }
        return "<" + boost::lexical_cast<std::string>(level) + ">";
    }

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: prepend locality prefix 
    struct locality_prefix 
      : boost::logging::formatter::class_<
            locality_prefix, 
            boost::logging::formatter::implement_op_equal::no_context
        > 
    {
        boost::uint32_t prefix_;

        locality_prefix(naming::id_type const& prefix) 
          : prefix_(naming::get_prefix_from_id(prefix)) 
        {}

        void operator()(param str) const 
        {
            std::stringstream out;
            out << "(" << std::hex << std::setw(4) << std::setfill('0') 
                << prefix_ << ") ";
            str.prepend_string(out.str());
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom log destination: send generated strings to console
    struct console : boost::logging::destination::is_generic 
    {
        console(naming::id_type const& prefix, int level, 
                components::server::logging_destination dest)
          : prefix_(prefix), level_(level), dest_(dest)
        {}

        template<typename MsgType> 
        void operator()(MsgType const& msg) const 
        {
            components::console_logging(prefix_, dest_, level_, msg);
        }

        bool operator==(console const& rhs) const 
        { 
            return prefix_ == rhs.prefix_ && dest_ == rhs.dest_; 
        }

        naming::id_type prefix_;
        int level_;
        components::server::logging_destination dest_;
    };


    ///////////////////////////////////////////////////////////////////////////
    // this is required in order to use the logging library
    BOOST_DEFINE_LOG_FILTER_WITH_ARGS(agas_level, filter_type, 
        boost::logging::level::disable_all) 
    BOOST_DEFINE_LOG(agas_logger, logger_type) 

    // initialize logging for AGAS
    void init_agas_log(util::section const& ini, bool isconsole, 
        naming::id_type const& console_prefix, naming::id_type const& prefix) 
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.agas")) {
            util::section const* logini = ini.get_section("hpx.logging.agas");
            BOOST_ASSERT(NULL != logini);

            loglevel = logini->get_entry("level", "");
            logdest = logini->get_entry("destination", "");
            logformat = detail::unescape(logini->get_entry("format", ""));
        }

        int lvl = boost::logging::level::disable_all;
        if (!loglevel.empty()) 
            lvl = detail::get_log_level(loglevel);

        if (boost::logging::level::disable_all != lvl) 
        {
            if (logdest.empty())      // ensure minimal defaults
                logdest = isconsole ? "cerr" : "console";
            if (logformat.empty())
                logformat = "|\\n";

            agas_logger()->writer().add_destination("console",
                console(console_prefix, lvl, components::server::destination_agas));
            agas_logger()->writer().write(logformat, logdest);
            agas_logger()->writer().add_formatter(locality_prefix(prefix));
            agas_logger()->mark_as_initialized();
            agas_level()->set_enabled(detail::get_log_level(loglevel));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // this is required in order to use the logging library for timings
    BOOST_DEFINE_LOG_FILTER_WITH_ARGS(timing_level, filter_type, 
        boost::logging::level::disable_all) 
    BOOST_DEFINE_LOG(timing_logger, logger_type) 

    // initialize logging for performance measurements
    void init_timing_log(util::section const& ini, bool isconsole, 
        naming::id_type const& console_prefix, naming::id_type const& prefix) 
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.timing")) {
            util::section const* logini = ini.get_section("hpx.logging.timing");
            BOOST_ASSERT(NULL != logini);

            loglevel = logini->get_entry("level", "");
            logdest = logini->get_entry("destination", "");
            logformat = detail::unescape(logini->get_entry("format", ""));
        }

        int lvl = boost::logging::level::disable_all;
        if (!loglevel.empty()) 
            lvl = detail::get_log_level(loglevel);

        if (boost::logging::level::disable_all != lvl) 
        {
            if (logdest.empty())      // ensure minimal defaults
                logdest = isconsole ? "cerr" : "console";
            if (logformat.empty())
                logformat = "|\\n";

            timing_logger()->writer().add_destination("console",
                console(console_prefix, lvl, components::server::destination_timing));
            timing_logger()->writer().write(logformat, logdest);
            timing_logger()->writer().add_formatter(locality_prefix(prefix));
            timing_logger()->mark_as_initialized();
            timing_level()->set_enabled(detail::get_log_level(loglevel));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    BOOST_DEFINE_LOG_FILTER_WITH_ARGS(hpx_level, filter_type,
        boost::logging::level::disable_all) 
    BOOST_DEFINE_LOG(hpx_logger, logger_type) 

    BOOST_DEFINE_LOG_FILTER_WITH_ARGS(hpx_error_level, filter_type,
        boost::logging::level::fatal) 
    BOOST_DEFINE_LOG(hpx_error_logger, logger_type)

    // initialize logging for HPX runtime
    void init_hpx_logs(util::section const& ini, bool isconsole, 
        naming::id_type const& console_prefix, naming::id_type const& prefix) 
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging")) {
            util::section const* logini = ini.get_section("hpx.logging");
            BOOST_ASSERT(NULL != logini);

            loglevel = logini->get_entry("level", "");
            logdest = logini->get_entry("destination", "");
            logformat = detail::unescape(logini->get_entry("format", ""));
        }

        int lvl = boost::logging::level::disable_all;
        if (!loglevel.empty()) 
            lvl = detail::get_log_level(loglevel);

        if (logdest.empty())      // ensure minimal defaults
            logdest = isconsole ? "cerr" : "console";
        if (logformat.empty())
            logformat = "|\\n";

        if (boost::logging::level::disable_all != lvl) {
            hpx_logger()->writer().add_destination("console",
                console(console_prefix, lvl, components::server::destination_hpx));
            hpx_logger()->writer().write(logformat, logdest);
            hpx_logger()->writer().add_formatter(locality_prefix(prefix));
            hpx_logger()->mark_as_initialized();
            hpx_level()->set_enabled(lvl);

            // errors are logged to the given destination and to cerr
            hpx_error_logger()->writer().add_destination("console",
                console(console_prefix, lvl, components::server::destination_hpx));
            hpx_error_logger()->writer().write(logformat, logdest + " cerr");
            hpx_error_logger()->writer().add_formatter(locality_prefix(prefix));
            hpx_error_logger()->mark_as_initialized();
            hpx_error_level()->set_enabled(lvl);
        }
        else {
            // errors are always logged to cerr 
            if (!isconsole) {
                hpx_error_logger()->writer().add_destination("console",
                    console(console_prefix, lvl, components::server::destination_hpx));
                hpx_error_logger()->writer().write(logformat, "console");
            }
            else {
                hpx_error_logger()->writer().write(logformat, "cerr");
            }
            hpx_error_logger()->writer().add_formatter(locality_prefix(prefix));
            hpx_error_logger()->mark_as_initialized();
            hpx_error_level()->set_enabled(boost::logging::level::fatal);
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

            loglevel = logini->get_entry("level", "");
            logdest = logini->get_entry("destination", "");
            logformat = detail::unescape(logini->get_entry("format", ""));
        }

        int lvl = boost::logging::level::disable_all;
        if (!loglevel.empty()) 
            lvl = detail::get_log_level(loglevel);

        if (boost::logging::level::disable_all != lvl) 
        {
            if (logdest.empty())      // ensure minimal defaults
                logdest = "cerr";
            if (logformat.empty())
                logformat = "|\\n";

            agas_console_logger()->writer().write(logformat, logdest);
            agas_console_logger()->mark_as_initialized();
            agas_console_level()->set_enabled(detail::get_log_level(loglevel));
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

            loglevel = logini->get_entry("level", "");
            logdest = logini->get_entry("destination", "");
            logformat = detail::unescape(logini->get_entry("format", ""));
        }

        int lvl = boost::logging::level::disable_all;
        if (!loglevel.empty()) 
            lvl = detail::get_log_level(loglevel);

        if (boost::logging::level::disable_all != lvl) 
        {
            if (logdest.empty())      // ensure minimal defaults
                logdest = "cerr";
            if (logformat.empty())
                logformat = "|\\n";

            timing_console_logger()->writer().write(logformat, logdest);
            timing_console_logger()->mark_as_initialized();
            timing_console_level()->set_enabled(detail::get_log_level(loglevel));
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

            loglevel = logini->get_entry("level", "");
            logdest = logini->get_entry("destination", "");
            logformat = detail::unescape(logini->get_entry("format", ""));
        }

        int lvl = boost::logging::level::disable_all;
        if (!loglevel.empty()) 
            lvl = detail::get_log_level(loglevel);

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
#define HPX_TIMEFORMAT "$hh:$mm.$ss"
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
                    "format = ${HPX_LOGFORMAT:%time%(" HPX_TIMEFORMAT ") [%idx%]|\\n}",

                // general console logging
                    "[hpx.logging.console]",
                    "level = ${HPX_LOGLEVEL:$[hpx.logging.level]}",
                    "destination = ${HPX_CONSOLE_LOGDESTINATION:file(hpx.$[system.pid].log)}",
                    "format = ${HPX_CONSOLE_LOGFORMAT:|}",

                // logging related to timing
                    "[hpx.logging.timing]",
                    "level = ${HPX_TIMING_LOGLEVEL:0}",
                    "destination = ${HPX_TIMING_LOGDESTINATION:console}",
                    "format = ${HPX_TIMING_LOGFORMAT:%time%(" HPX_TIMEFORMAT ") [%idx%] [TIM] |\\n}",

                // console logging related to timing
                    "[hpx.logging.console.timing]",
                    "level = ${HPX_TIMING_LOGLEVEL:$[hpx.logging.timing.level]}",
                    "destination = ${HPX_CONSOLE_TIMING_LOGDESTINATION:file(hpx.timing.$[system.pid].log)}",
                    "format = ${HPX_CONSOLE_TIMING_LOGFORMAT:|}",

                // logging related to AGAS
                    "[hpx.logging.agas]",
                    "level = ${HPX_AGAS_LOGLEVEL:0}",
                    "destination = ${HPX_AGAS_LOGDESTINATION:console}",
                    "format = ${HPX_AGAS_LOGFORMAT:%time%(" HPX_TIMEFORMAT ") [%idx%][AGAS] |\\n}",

                // console logging related to AGAS
                    "[hpx.logging.console.agas]",
                    "level = ${HPX_AGAS_LOGLEVEL:$[hpx.logging.agas.level]}",
                    "destination = ${HPX_CONSOLE_AGAS_LOGDESTINATION:file(hpx.agas.$[system.pid].log)}",
                    "format = ${HPX_CONSOLE_AGAS_LOGFORMAT:|}"
                ;
        }
        catch (std::exception const&) {
            // just in case something goes wrong
            std::cerr << "caught std::exception during initialization" 
                      << std::endl;
        }
    }

    struct init_logging_tag {};
    std::vector<std::string> const& get_logging_data()
    {
        static_<logging_configuration, init_logging_tag> init;
        return init.get().prefill_;
    }

    ///////////////////////////////////////////////////////////////////////////
    #define HPX_MAX_GETCONSOLEPREFIX_RETRIES    10
    #define HPX_SLEEP_GETCONSOLEPREFIX_RETRIES  100   // [ms]

    init_logging::init_logging(runtime_configuration& ini, bool isconsole,
        naming::resolver_client& agas_client, naming::id_type const& prefix)
    {
        naming::id_type console_prefix;
        for (int i = 0; i < HPX_MAX_GETCONSOLEPREFIX_RETRIES; ++i)
        {
            if (agas_client.get_console_prefix(console_prefix))
                break;

            boost::this_thread::sleep(boost::get_system_time() + 
                boost::posix_time::milliseconds(HPX_SLEEP_GETCONSOLEPREFIX_RETRIES));
        }

        init_agas_log(ini, isconsole, console_prefix, prefix);
        init_timing_log(ini, isconsole, console_prefix, prefix);
        init_hpx_logs(ini, isconsole, console_prefix, prefix);

        // initialize console logs (if appropriate)
        if (isconsole) {
            init_agas_console_log(ini);
            init_timing_console_log(ini);
            init_hpx_console_log(ini);
        }
    }

///////////////////////////////////////////////////////////////////////////////
}}}


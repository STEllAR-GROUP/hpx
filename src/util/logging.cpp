//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstdlib>
#include <hpx/util/logging.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <boost/version.hpp>
#include <boost/config.hpp>
#include <boost/filesystem/operations.hpp>

#include <boost/assign/std/vector.hpp>
#include <boost/logging/format/named_write.hpp>

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
    // this is required in order to use the logging library
    BOOST_DEFINE_LOG_FILTER_WITH_ARGS(agas_level, filter_type, 
        boost::logging::level::disable_all) 
    BOOST_DEFINE_LOG(agas_logger, logger_type) 

    // initialize logging for AGAS
    void init_agas_logs(util::section const& ini) 
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.agas.logging")) {
            util::section const* logini = ini.get_section("hpx.agas.logging");
            BOOST_ASSERT(NULL != logini);

            loglevel = logini->get_entry("level");
            logdest = logini->get_entry("destination");
            logformat = detail::unescape(logini->get_entry("format"));
        }

        if (!loglevel.empty() && 
            boost::logging::level::disable_all != detail::get_log_level(loglevel)) 
        {
            agas_logger()->writer().write(logformat, logdest);
            agas_logger()->mark_as_initialized();
            agas_level()->set_enabled(boost::logging::level::fatal);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // this is required in order to use the logging library for timings
    BOOST_DEFINE_LOG_FILTER_WITH_ARGS(timing_level, filter_type, 
        boost::logging::level::disable_all) 
    BOOST_DEFINE_LOG(timing_logger, logger_type) 

    // initialize logging for performance measurements
    void init_timing_logs(util::section const& ini) 
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.timing.logging")) {
            util::section const* logini = ini.get_section("hpx.timing.logging");
            BOOST_ASSERT(NULL != logini);

            loglevel = logini->get_entry("level");
            logdest = logini->get_entry("destination");
            logformat = detail::unescape(logini->get_entry("format"));
        }

        if (!loglevel.empty() && 
            boost::logging::level::disable_all != detail::get_log_level(loglevel)) 
        {
            timing_logger()->writer().write(logformat, logdest);
            timing_logger()->mark_as_initialized();
            timing_level()->set_enabled(detail::get_log_level(loglevel));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    BOOST_DEFINE_LOG_FILTER_WITH_ARGS(hpx_level, filter_type,
        boost::logging::level::disable_all) 
    BOOST_DEFINE_LOG(hpx_logger, logger_type) 

    // initialize logging for HPX runtime
    void init_hpx_logs(util::section const& ini) 
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging")) {
            util::section const* logini = ini.get_section("hpx.logging");
            BOOST_ASSERT(NULL != logini);

            loglevel = logini->get_entry("level");
            logdest = logini->get_entry("destination");
            logformat = detail::unescape(logini->get_entry("format"));
        }

        if (!loglevel.empty() && 
            boost::logging::level::disable_all != detail::get_log_level(loglevel)) 
        {
            hpx_logger()->writer().write(logformat, logdest);
            hpx_logger()->mark_as_initialized();
            hpx_level()->set_enabled(detail::get_log_level(loglevel));
        }
        else {
        // if logging is disabled re-route error messages to cerr
            hpx_logger()->writer().write(logformat, "cerr");
            hpx_logger()->mark_as_initialized();
            hpx_level()->set_enabled(detail::get_log_level(loglevel));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    struct init_logging
    {
        init_logging()
        {
            try {
                // add default logging configuration as defaults to the ini data
                // this will be overwritten by related entries in the read hpx.ini
                using namespace boost::assign;
                std::vector<std::string> prefill; 
                prefill +=
                        "[hpx.logging]",
                        "level = ${HPX_LOGLEVEL:0}",
                        "destination = ${HPX_LOGDESTINATION:file(hpx.$[system.pid].log)}",
#if defined(BOOST_WINDOWS)
                        "format = ${HPX_LOGFORMAT:%time%($hh:$mm.$ss.$mili) [%idx%]|\\n}",
#else
                        // the Boost.Log generates bogus milliseconds on Linux systems
                        "format = ${HPX_LOGFORMAT:%time%($hh:$mm.$ss) [%idx%]|\\n}",
#endif

                        "[hpx.timing.logging]",
                        "level = ${HPX_TIMING_LOGLEVEL:0}",
                        "destination = ${HPX_TIMING_LOGDESTINATION:file(hpx.timing.$[system.pid].log)}",
#if defined(BOOST_WINDOWS)
                        "format = ${HPX_TIMING_LOGFORMAT:%time%($hh:$mm.$ss.$mili) [%idx%] [TIM] |\\n}",
#else
                        // the Boost.Log generates bogus milliseconds on Linux systems
                        "format = ${HPX_TIMING_LOGFORMAT:%time%($hh:$mm.$ss) [%idx%] [TIM] |\\n}",
#endif

                        "[hpx.agas.logging]",
                        "level = ${HPX_AGAS_LOGLEVEL:0}",
                        "destination = ${HPX_AGAS_LOGDESTINATION:file(hpx.agas.$[system.pid].log)}",
#if defined(BOOST_WINDOWS)
                        "format = ${HPX_AGAS_LOGFORMAT:%time%($hh:$mm.$ss.$mili) [%idx%][AGAS] |\\n}"
#else
                        // the Boost.Log generates bogus milliseconds on Linux systems
                        "format = ${HPX_AGAS_LOGFORMAT:%time%($hh:$mm.$ss) [%idx%][AGAS] |\\n}"
#endif
                    ;

                // initialize logging 
                util::runtime_configuration ini(prefill);
                util::init_agas_logs(ini);
                util::init_timing_logs(ini);
                util::init_hpx_logs(ini);
            }
            catch (std::exception const&) {
                // just in case something goes wrong
                std::cerr << "caught std::exception during initialization" 
                          << std::endl;
            }
        }
    };
    init_logging const init_logging_;

///////////////////////////////////////////////////////////////////////////////
}}

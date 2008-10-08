//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstdlib>
#include <hpx/util/logging.hpp>
#include <boost/logging/format/named_write.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util 
{
    namespace detail
    {
        int get_log_level(char const* env)
        {
            try {
                int env_val = boost::lexical_cast<int>(env);
                if (env_val <= 0)
                    return -1;      // disable all
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
                return 0;
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // this is required in order to use the logging library
    BOOST_DEFINE_LOG_FILTER(dgas_level, filter_type) 
    BOOST_DEFINE_LOG(dgas_logger, logger_type) 

    // initialize logging for DGAS
    void init_dgas_logs() 
    {
        using namespace std;    // some systems have getenv in namespace std
        char const* dgas_level_env = getenv("HPX_DGAS_LOGLEVEL");
        if (NULL != dgas_level_env) 
        {
            char const* logdest = getenv("HPX_DGAS_LOGDESTINATION");
            if (NULL == logdest)
                logdest = "file(dgas.log)";

            char const* logformat = getenv("HPX_DGAS_LOGFORMAT");
            if (NULL == logformat)
                logformat = "%time%($hh:$mm.$ss.$mili) [%idx%][DGAS] |\n";

            // formatting    : time [idx][DGAS] message \n
            // destinations  : file "dgas.log"
            dgas_logger()->writer().write(logformat, logdest);
            dgas_logger()->mark_as_initialized();
            dgas_level()->set_enabled(detail::get_log_level(dgas_level_env));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    BOOST_DEFINE_LOG_FILTER(hpx_level, filter_type) 
    BOOST_DEFINE_LOG(hpx_logger, logger_type) 

    // initialize logging for threadmanager
    void init_hpx_logs() 
    {
        using namespace std;    // some systems have getenv in namespace std
        char const* hpx_level_env = getenv("HPX_LOGLEVEL");
        if (NULL != hpx_level_env) 
        {
            char const* logdest = getenv("HPX_LOGDESTINATION");
            if (NULL == logdest)
                logdest = "file(hpx.log)";

            char const* logformat = getenv("HPX_LOGFORMAT");
            if (NULL == logformat)
                logformat = "%time%($hh:$mm.$ss.$mili) [%idx%]|\n";

            // formatting    : time [idx][...] message \n
            // destinations  : file "hpx.log"
            hpx_logger()->writer().write(logformat, logdest);
            hpx_logger()->mark_as_initialized();
            hpx_level()->set_enabled(detail::get_log_level(hpx_level_env));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    struct init_logging
    {
        init_logging()
        {
            util::init_dgas_logs();
            util::init_hpx_logs();
        }
    };
    init_logging const init_logging_;

///////////////////////////////////////////////////////////////////////////////
}}

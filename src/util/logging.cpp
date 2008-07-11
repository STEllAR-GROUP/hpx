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
                logformat = "%time%($hh:$mm.$ss.$mili) [DGAS][%idx%] |\n";
                
            // formatting    : time [DGAS][idx] message \n
            // destinations  : file "dgas.log"
            dgas_logger()->writer().write(logformat, logdest);
            dgas_logger()->mark_as_initialized();
            dgas_level()->set_enabled(detail::get_log_level(dgas_level_env));
        }
    }

    BOOST_DEFINE_LOG_FILTER(osh_level, filter_type) 
    BOOST_DEFINE_LOG(osh_logger, logger_type) 

    // initialize logging for one size heaps
    void init_onesizeheap_logs() 
    {
        using namespace std;    // some systems have getenv in namespace std
        char const* osh_level_env = getenv("HPX_OSH_LOGLEVEL");
        if (NULL != osh_level_env) 
        {
            char const* logdest = getenv("HPX_OSH_LOGDESTINATION");
            if (NULL == logdest)
                logdest = "file(osh.log)";
                
            char const* logformat = getenv("HPX_OSH_LOGFORMAT");
            if (NULL == logformat)
                logformat = "%time%($hh:$mm.$ss.$mili) [OSH][%idx%] |\n";
                
            // formatting    : time [OSH][idx] message \n
            // destinations  : file "osh.log"
            osh_logger()->writer().write(logformat, logdest);
            osh_logger()->mark_as_initialized();
            osh_level()->set_enabled(detail::get_log_level(osh_level_env));
        }
    }

    BOOST_DEFINE_LOG_FILTER(tm_level, filter_type) 
    BOOST_DEFINE_LOG(tm_logger, logger_type) 

    // initialize logging for threadmanager
    void init_threadmanager_logs() 
    {
        using namespace std;    // some systems have getenv in namespace std
        char const* tm_level_env = getenv("HPX_TM_LOGLEVEL");
        if (NULL != tm_level_env) 
        {
            char const* logdest = getenv("HPX_TM_LOGDESTINATION");
            if (NULL == logdest)
                logdest = "file(tm.log)";
                
            char const* logformat = getenv("HPX_TM_LOGFORMAT");
            if (NULL == logformat)
                logformat = "%time%($hh:$mm.$ss.$mili) [TM][%idx%] |\n";
                
            // formatting    : time [TM][idx] message \n
            // destinations  : file "tm.log"
            tm_logger()->writer().write(logformat, logdest);
            tm_logger()->mark_as_initialized();
            tm_level()->set_enabled(detail::get_log_level(tm_level_env));
        }
    }

///////////////////////////////////////////////////////////////////////////////
}}

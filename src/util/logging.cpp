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
    // this is required in order to use the logging library
    BOOST_DEFINE_LOG_FILTER(dgas_level, filter_type) 
    BOOST_DEFINE_LOG(dgas_logger, logger_type) 

    // initialize logging for DGAS
    void init_dgas_logs() 
    {
        using namespace std;    // some systems have getenv in namespace std
        if (NULL != getenv("HPX_DGAS_LOGLEVEL")) 
        {
            char const* logdest = getenv("HPX_DGAS_LOGDESTINATION");
            if (NULL == logdest)
                logdest = "cout file(dgas.log)";
                
            char const* logformat = getenv("HPX_DGAS_LOGFORMAT");
            if (NULL == logformat)
                logformat = "%time%($hh:$mm.$ss.$mili) [DGAS][%idx%] |\n";
                
            // formatting    : time [DGAS][idx] message \n
            // destinations  : console, file "dgas.log"
            dgas_logger()->writer().write(logformat, logdest);
            dgas_logger()->mark_as_initialized();
        }
    }

    BOOST_DEFINE_LOG_FILTER(osh_level, filter_type) 
    BOOST_DEFINE_LOG(osh_logger, logger_type) 

    // initialize logging for one size heaps
    void init_onesizeheap_logs() 
    {
        using namespace std;    // some systems have getenv in namespace std
        if (NULL != getenv("HPX_OSH_LOGLEVEL")) 
        {
            char const* logdest = getenv("HPX_OSH_LOGDESTINATION");
            if (NULL == logdest)
                logdest = "cout file(osh.log)";
                
            char const* logformat = getenv("HPX_OSH_LOGFORMAT");
            if (NULL == logformat)
                logformat = "%time%($hh:$mm.$ss.$mili) [OSH][%idx%] |\n";
                
            // formatting    : time [OSH][idx] message \n
            // destinations  : console, file "osh.log"
            osh_logger()->writer().write(logformat, logdest);
            osh_logger()->mark_as_initialized();
        }
    }

    BOOST_DEFINE_LOG_FILTER(tm_level, filter_type) 
    BOOST_DEFINE_LOG(tm_logger, logger_type) 

    // initialize logging for threadmanager
    void init_threadmanager_logs() 
    {
        using namespace std;    // some systems have getenv in namespace std
        if (NULL != getenv("HPX_TM_LOGLEVEL")) 
        {
            char const* logdest = getenv("HPX_TM_LOGDESTINATION");
            if (NULL == logdest)
                logdest = "file(tm.log)";
                
            char const* logformat = getenv("HPX_TM_LOGFORMAT");
            if (NULL == logformat)
                logformat = "%time%($hh:$mm.$ss.$mili) [TM][%idx%] |\n";
                
            // formatting    : time [TM][idx] message \n
            // destinations  : console, file "tm.log"
            tm_logger()->writer().write(logformat, logdest);
            tm_logger()->mark_as_initialized();
        }
    }

///////////////////////////////////////////////////////////////////////////////
}}

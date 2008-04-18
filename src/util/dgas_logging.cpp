//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstdlib>
#include <hpx/util/dgas_logging.hpp>
#include <boost/logging/format/named_write.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util 
{
    // this is required in order to use the logging library
    BOOST_DEFINE_LOG_FILTER(g_l_level, filter_type) 
    BOOST_DEFINE_LOG(g_l, logger_type) 

    // initialize logging for DGAS
    void init_dgas_logs() 
    {
        using namespace std;    // some systems have getenv in namespace std
        if (NULL != getenv("HPX_LOGLEVEL")) 
        {
            char const* logdest = getenv("HPX_DGAS_LOGDESTINATION");
            if (NULL != logdest)
                logdest = "cout file(dgas.log)";
                
            char const* logformat = getenv("HPX_DGAS_LOGFORMAT");
            if (NULL == logformat)
                logformat = "%time%($hh:$mm.$ss.$mili) [DGAS][%idx%] |\n";
                
            // formatting    : time [DGAS][idx] message \n
            // destinations  : console, file "dgas.log"
            g_l()->writer().write(logformat, logdest);
            g_l()->mark_as_initialized();
        }
    }

///////////////////////////////////////////////////////////////////////////////
}}

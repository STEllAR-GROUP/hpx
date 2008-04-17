//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

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
        // formatting    : time [DGAS][idx] message \n
        // destinations  : console, file "dgas.log"
        g_l()->writer().write("%time%($hh:$mm.$ss.$mili) [DGAS][%idx%] |\n", 
            "cout file(dgas.log)");
        g_l()->mark_as_initialized();
    }

///////////////////////////////////////////////////////////////////////////////
}}

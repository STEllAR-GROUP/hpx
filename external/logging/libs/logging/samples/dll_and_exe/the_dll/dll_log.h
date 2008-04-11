/**
 Boost Logging library

 Author: John Torjo, www.torjo.com

 Copyright (C) 2007 John Torjo (see www.torjo.com for email)

 Distributed under the Boost Software License, Version 1.0.
    (See accompanying file LICENSE_1_0.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)

 See http://www.boost.org for updates, documentation, and revision history.
 See http://www.torjo.com/log2/ for more details
*/

#ifndef THE_DLL_log_H
#define THE_DLL_log_H

#pragma once
#include "the_dll.h"


#include <boost/logging/format.hpp>

using namespace boost::logging::scenario::usage;
typedef use<
        filter_::change::single_thread, 
        filter_::level::no_levels, 
        logger_::change::single_thread, 
        logger_::favor::correctness> finder;

#ifdef THE_DLL_EXPORTS
// internally

// note : we export this filter & logger
THE_DLL_API BOOST_DECLARE_LOG_FILTER(g_dll_log_filter, finder::filter ) 
THE_DLL_API BOOST_DECLARE_LOG(g_dll_l, finder::logger)

#else
// what we expose to our clients

/* 
Equivalent to:

THE_DLL_API BOOST_DECLARE_LOG_FILTER(g_dll_log_filter, finder::filter ) 
THE_DLL_API BOOST_DECLARE_LOG(g_dll_l, finder::logger)

However, we want to clearly specify the DLL's option: either BOOST_LOG_COMPILE_FAST_ON or BOOST_LOG_COMPILE_FAST_OFF
(in our case, BOOST_LOG_COMPILE_FAST_OFF).

We need to clearly specify this to our client which will also be using our log, but might have different settings.
For instance, he could have a different gather_msg, a different BOOST_LOG_COMPILE_* setting, a different type of log, etc.
*/
THE_DLL_API finder::filter * g_dll_log_filter();
THE_DLL_API finder::logger * g_dll_l();
#endif

#define L_DLL_ BOOST_LOG_USE_LOG_IF_FILTER(g_dll_l(), g_dll_log_filter()->is_enabled() ) 


// in this function, we write some log messages (from the DLL)
THE_DLL_API void write_to_dll_logs();

#endif

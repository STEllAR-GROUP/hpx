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

// my_app_log.h - DECLARE your loggers & filters here
#ifndef my_app_LOG_H_header
#define my_app_LOG_H_header

#include <boost/logging/format/named_write_fwd.hpp>
// #include <boost/logging/writer/on_dedicated_thread.hpp> // uncomment if you want to do logging on a dedicated thread

namespace bl = boost::logging;
typedef bl::named_logger<>::type logger_type;
typedef bl::level::holder filter_type;

BOOST_DECLARE_LOG_FILTER(g_l_level, filter_type)
BOOST_DECLARE_LOG(g_l, logger_type)

#define L_(lvl) BOOST_LOG_USE_LOG_IF_LEVEL(g_l(), g_l_level(), lvl )

// initialize thy logs..
void init_logs();

#endif

// my_app_log.cpp - DEFINE your loggers & filters here
#include "my_app_log.h"
#include <boost/logging/format/named_write.hpp>

BOOST_DEFINE_LOG_FILTER(g_l_level, filter_type ) 
BOOST_DEFINE_LOG(g_l, logger_type) 


void init_logs() {
    // formatting    : time [idx] message \n
    // destinations  : console, file "out.txt" and debug window
    g_l()->writer().write("%time%($hh:$mm.$ss.$mili) [%idx%] |\n", "cout file(out.txt) debug");
    g_l()->mark_as_initialized();
}

void use_logger() {
    int i = 1;
    L_(debug) << "this is a simple message " << i;
    std::string hello = "hello";
    L_(info) << hello << " world";
}

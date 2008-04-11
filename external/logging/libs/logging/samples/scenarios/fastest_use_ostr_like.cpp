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

/**
@example fastest_use_ostr_like.cpp

@copydoc fastest_use_ostr_like

@page fastest_use_ostr_like fastest_use_ostr_like.cpp Example


This usage:
- Fastest. It does not use formatters and destinations (thus, 
  the flexibility that comes with formatters/destinations) is gone
- You have one filter, which can be turned on or off
- You have 2 loggers: app and err.
- The app writes to console, the err writes to "err.txt" file
- It uses the << operator as the logging syntax

Here's what the output will be:

The console:
@code
this is so cool 1
this is so cool again 2
hello, world
good to be back ;) 4
@endcode

The err.txt file:
@code
first error 3
second error 5
@endcode


*/



#define BOOST_LOG_COMPILE_FAST_OFF
#include <boost/logging/logging.hpp>
#include <boost/logging/format.hpp>

using namespace boost::logging;

typedef logger< default_, destination::cout> app_log_type;
typedef logger< default_, destination::file> err_log_type;

BOOST_DEFINE_LOG_FILTER(g_log_filter, filter::no_ts )

BOOST_DEFINE_LOG(g_log_app, app_log_type )
BOOST_DEFINE_LOG_WITH_ARGS( g_log_err, err_log_type, ("err.txt") )

#define LAPP_ BOOST_LOG_USE_LOG_IF_FILTER(g_log_app(), g_log_filter()->is_enabled() ) 
#define LERR_ BOOST_LOG_USE_LOG_IF_FILTER(g_log_err(), g_log_filter()->is_enabled() ) 

void fastest_use_ostr_like_example() {
    g_log_app()->mark_as_initialized();
    g_log_err()->mark_as_initialized();

    int i = 1;
    LAPP_ << "this is so cool " << i++ << "\n";
    LAPP_ << "this is so cool again " << i++ << "\n";
    LERR_ << "first error " << i++ << "\n";

    std::string hello = "hello", world = "world";
    LAPP_ << hello << ", " << world << "\n";

    g_log_filter()->set_enabled(false);
    LAPP_ << "this will not be written to the log";
    LAPP_ << "this won't be written to the log";
    LERR_ << "this error is not logged " << i++;

    g_log_filter()->set_enabled(true);
    LAPP_ << "good to be back ;) " << i++ << "\n";
    LERR_ << "second error " << i++ << "\n";
}




int main() {
    fastest_use_ostr_like_example();
}


// End of file


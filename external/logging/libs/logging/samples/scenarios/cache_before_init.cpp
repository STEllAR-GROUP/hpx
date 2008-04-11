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
@example cache_before_init.cpp

@copydoc cache_before_init 

@page cache_before_init cache_before_init.cpp Example


This usage:
- You log a few messages before initializing the logs
- You use one filter, based on levels
- You specify a certain level for the filter, so that not all of the messages should be logged
- You turn the cache off, and only those messages matching the filter are logged

Optimizations:
- use a cache string (from optimize namespace), in order to make formatting the message faster

Logs:
- Error messages go into err.txt file
  - formatting - prefix each message by time, index, and append newline
- Info output goes to console, and a file called out.txt
  - formatting - prefix each message by "[app]", time, and append newline
- Debug messages go to the debug output window, and a file called out.txt
  - formatting - prefix each message by "[dbg]", time, and append newline


Here's how the output will look like:

The debug output window:
@code
17:29.41 [dbg] some debug message after logs have been initialized 11
@endcode


The console:
@code
17:29.41 [app] hello, world
17:29.41 [app] coolio 4
17:29.41 [app] after logs have been initialized 10
@endcode


The out.txt file:
@code
17:29.41 [app] hello, world
17:29.41 [app] coolio 4
17:29.41 [app] after logs have been initialized 10
17:29.41 [dbg] some debug message after logs have been initialized 11
@endcode


The err.txt file
@code
17:29.41 [1] first error 3
17:29.41 [2] second error 5
@endcode
*/


#define BOOST_LOG_BEFORE_INIT_USE_CACHE_FILTER

// uncomment this, and all messages inside singleton's constructor will be logged!
//#define BOOST_LOG_BEFORE_INIT_LOG_ALL

// uncomment this, and NO messages inside singleton's constructor will be logged
//#define BOOST_LOG_BEFORE_INIT_IGNORE_BEFORE_INIT

#include <boost/logging/format_fwd.hpp>

BOOST_LOG_FORMAT_MSG( optimize::cache_string_one_str<> )

#include <boost/logging/format.hpp>

typedef boost::logging::logger_format_write< > logger_type;


BOOST_DECLARE_LOG_FILTER(g_log_level, boost::logging::level::holder ) // holds the application log level
BOOST_DECLARE_LOG(g_log_err, logger_type) 
BOOST_DECLARE_LOG(g_log_app, logger_type)
BOOST_DECLARE_LOG(g_log_dbg, logger_type)

#define LDBG_ BOOST_LOG_USE_LOG_IF_LEVEL(g_log_dbg(), g_log_level(), debug ) << "[dbg] "
#define LERR_ BOOST_LOG_USE_LOG_IF_LEVEL(g_log_err(), g_log_level(), error )
#define LAPP_ BOOST_LOG_USE_LOG_IF_LEVEL(g_log_app(), g_log_level(), info ) << "[app] "

BOOST_DEFINE_LOG_FILTER(g_log_level, boost::logging::level::holder ) 
BOOST_DEFINE_LOG(g_log_err, logger_type)
BOOST_DEFINE_LOG(g_log_app, logger_type)
BOOST_DEFINE_LOG(g_log_dbg, logger_type)

using namespace boost::logging;

struct singleton {
    singleton() {
        // note: these messages are written before logs are initialized
        int i = 1;
        LDBG_ << "this is so cool " << i++;
        LDBG_ << "this is so cool again " << i++;
        LERR_ << "first error " << i++;

        std::string hello = "hello", world = "world";
        LAPP_ << hello << ", " << world;

        LAPP_ << "coolio " << i++;
        LERR_ << "second error " << i++;
        LDBG_ << "some debug message" << i++;
    }
} s_;

void init_logs() {
    // Err log
    g_log_err()->writer().add_formatter( formatter::idx(), "[%] "  );
    g_log_err()->writer().add_formatter( formatter::time("$hh:$mm.$ss ") );
    g_log_err()->writer().add_formatter( formatter::append_newline() );
    g_log_err()->writer().add_destination( destination::file("err.txt") );

    destination::file out("out.txt");
    // App log
    g_log_app()->writer().add_formatter( formatter::time("$hh:$mm.$ss ") );
    g_log_app()->writer().add_formatter( formatter::append_newline() );
    g_log_app()->writer().add_destination( out );
    g_log_app()->writer().add_destination( destination::cout() );

    // Debug log
    g_log_dbg()->writer().add_formatter( formatter::time("$hh:$mm.$ss ") );
    g_log_dbg()->writer().add_formatter( formatter::append_newline() );
    g_log_dbg()->writer().add_destination( out );
    g_log_dbg()->writer().add_destination( destination::dbg_window() );

    // if you change this, you'll get a different output (more or less verbose)
    g_log_level()->set_enabled(level::info);

    g_log_err()->mark_as_initialized();
    g_log_app()->mark_as_initialized();
    g_log_dbg()->mark_as_initialized();
}

void cache_before_init_example() {
    init_logs();
    int i = 10;
    LAPP_ << "after logs have been initialized " << i++;
    g_log_level()->set_enabled(level::debug);
    LDBG_ << "some debug message after logs have been initialized " << i++;
}




int main() {
    cache_before_init_example();
}


// End of file


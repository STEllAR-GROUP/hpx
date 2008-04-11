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
@example mul_levels_mul_logers.cpp

@copydoc mul_levels_mul_logers 

@page mul_levels_mul_logers mul_levels_mul_logers.cpp Example


This usage:
- You have multiple levels (in this example: debug < info < error)
- You want to format the message before it's written 
  (in this example: prefix it by time, by index and append newline to it)
- You have several loggers
- Each logger has several log destinations

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
07:52.30 [dbg] this is so cool 1
07:52.30 [dbg] this is so cool again 2
@endcode


The console:
@code
07:52.30 [app] hello, world
07:52.30 [app] good to be back ;) 4
@endcode


The out.txt file:
@code
07:52.30 [dbg] this is so cool 1
07:52.30 [dbg] this is so cool again 2
07:52.30 [app] hello, world
07:52.30 [app] good to be back ;) 4
@endcode


The err.txt file
@code
07:52.30 [1] first error 3
07:52.30 [2] second error 5
@endcode
*/



#include <boost/logging/format/named_write.hpp>
typedef boost::logging::named_logger<>::type logger_type;

#define LDBG_ BOOST_LOG_USE_LOG_IF_LEVEL(g_log_dbg(), g_log_level(), debug ) << "[dbg] "
#define LERR_ BOOST_LOG_USE_LOG_IF_LEVEL(g_log_err(), g_log_level(), error )
#define LAPP_ BOOST_LOG_USE_LOG_IF_LEVEL(g_log_app(), g_log_level(), info ) << "[app] "

BOOST_DEFINE_LOG_FILTER(g_log_level, boost::logging::level::holder ) 
BOOST_DEFINE_LOG(g_log_err, logger_type)
BOOST_DEFINE_LOG(g_log_app, logger_type)
BOOST_DEFINE_LOG(g_log_dbg, logger_type)

using namespace boost::logging;

void mul_levels_mul_logers_example() {
    // reuse the same destination for 2 logs
    destination::file out("out.txt");
    g_log_app()->writer().replace_destination("file", out);
    g_log_dbg()->writer().replace_destination("file", out);
    // formatting (first param) and destinations (second param)
    g_log_err()->writer().write("[%idx%] %time%($hh:$mm.$ss) |\n", "cout file(err.txt)"); // line A
    g_log_app()->writer().write("%time%($hh:$mm.$ss) |\n", "file cout");
    g_log_dbg()->writer().write("%time%($hh:$mm.$ss) |\n", "file cout debug");

    /* 
    Note : the "line A" above originally was:
    g_log_err()->writer().write("[%idx%] %time%($hh:$mm.$ss) |\n", "file(err.txt)");

    This caused a very strange assertion failure on Fedora8, when the program exits, while destroying the global variables.
    I've spent some time debugging it but to no avail. I will certainly look more into this.
    */

    g_log_app()->mark_as_initialized();
    g_log_err()->mark_as_initialized();
    g_log_dbg()->mark_as_initialized();


    int i = 1;
    LDBG_ << "this is so cool " << i++;
    LDBG_ << "this is so cool again " << i++;
    LERR_ << "first error " << i++;

    std::string hello = "hello", world = "world";
    LAPP_ << hello << ", " << world;

    g_log_level()->set_enabled(level::error);
    LDBG_ << "this will not be written anywhere";
    LAPP_ << "this won't be written anywhere either";

    g_log_level()->set_enabled(level::info);
    LAPP_ << "good to be back ;) " << i++;
    LERR_ << "second error " << i++;
}




int main() {
    mul_levels_mul_logers_example();
}


// End of file


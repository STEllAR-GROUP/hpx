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
@example ts_loger_one_filter.cpp

@copydoc ts_loger_one_filter

@page ts_loger_one_filter ts_loger_one_filter.cpp Example


This usage:
- You have one @b thread-safe logger
- You have one filter, which is always turned on
- You want to format the message before it's written 
- The logger has several log destinations
    - The output goes to console, debug output window, and a file called out.txt
    - Formatting - prefix each message by its index, thread id, and append newline

Optimizations:
- use a cache string (from optimize namespace), in order to make formatting the message faster

In this example, all output will be written to the console, debug window, and "out.txt" file.
It will look similar to:

@code
[T5884] [1] message 0
[T7168] [2] message 0
[T7932] [3] message 0
[T740] [4] message 0
[T8124] [5] message 0
[T5884] [6] message 1
[T5884] [7] message 2
[T740] [8] message 1
[T7168] [9] message 1
[T7932] [10] message 1
[T8124] [11] message 1
[T5884] [12] message 3
[T7168] [13] message 2
[T5884] [14] message 4
[T740] [15] message 2
[T7932] [16] message 2
[T8124] [17] message 2
[T7168] [18] message 3
[T5884] [19] message 5
...
@endcode

*/



#include <boost/logging/format_fwd.hpp>
BOOST_LOG_FORMAT_MSG( optimize::cache_string_one_str<> )

#include <boost/logging/format_ts.hpp>
#include <boost/logging/format/formatter/thread_id.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>

using namespace boost::logging;

typedef logger_format_write< default_, default_, writer::threading::ts_write > logger_type;

#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) 

BOOST_DEFINE_LOG_FILTER(g_log_filter, filter::no_ts ) 
BOOST_DEFINE_LOG(g_l, logger_type)

void do_sleep(int ms) {
    using namespace boost;
    xtime next;
    xtime_get( &next, TIME_UTC);
    next.nsec += (ms % 1000) * 1000000;

    int nano_per_sec = 1000000000;
    next.sec += next.nsec / nano_per_sec;
    next.sec += ms / 1000;
    next.nsec %= nano_per_sec;
    thread::sleep( next);
}

void use_log_thread() {
    for ( int i = 0; i < 50; ++i) {
        L_ << "message " << i ;
        do_sleep(1);
    }
}

void ts_logger_one_filter_example() {
    //         add formatters and destinations
    //         That is, how the message is to be formatted and where should it be written to
    g_l()->writer().add_formatter( formatter::idx(), "[%] "  );
    g_l()->writer().add_formatter( formatter::thread_id(), "[T%] "  );
    g_l()->writer().add_formatter( formatter::append_newline() );
    g_l()->writer().add_destination( destination::file("out.txt") );
    g_l()->writer().add_destination( destination::cout() );
    g_l()->writer().add_destination( destination::dbg_window() );
    g_l()->mark_as_initialized();

    for ( int i = 0 ; i < 5; ++i)
        boost::thread t( &use_log_thread);

    // allow for all threads to finish
    do_sleep( 5000);
}




int main() {
    ts_logger_one_filter_example();
}


// End of file


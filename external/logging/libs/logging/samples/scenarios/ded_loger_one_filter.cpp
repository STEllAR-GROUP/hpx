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
@example ded_loger_one_filter.cpp

@copydoc ded_loger_one_filter

@page ded_loger_one_filter ded_loger_one_filter.cpp Example


This usage:
- You have one @b thread-safe logger - the logging is done @ref boost::logging::writer::on_dedicated_thread "on a dedicated thread"
- You have one filter, which is always turned on
- You want to format the message before it's written 
- The logger has several log destinations
    - The output goes debug output window, and a file called out.txt
    - Formatting - prefix each message by time, its index, and append newline

Optimizations:
- use a cache string (from optimize namespace), in order to make formatting the message faster

In this example, all output will be written to the console, debug window, and "out.txt" file.
It will look similar to:

@code
...
30:33 [10] message 1
30:33 [11] message 2
30:33 [12] message 2
30:33 [13] message 2
30:33 [14] message 2
30:33 [15] message 3
30:33 [16] message 2
30:33 [17] message 3
30:33 [18] message 3
30:33 [19] message 4
30:33 [20] message 3
30:33 [21] message 3
30:33 [22] message 4
30:33 [23] message 4
30:33 [24] message 4
30:33 [25] message 4
30:33 [26] message 5
30:33 [27] message 5
30:33 [28] message 6
30:33 [29] message 6
30:33 [30] message 5
30:33 [31] message 5
30:33 [32] message 5
30:33 [33] message 6
30:33 [34] message 7
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

typedef logger_format_write< default_, default_, writer::threading::on_dedicated_thread > logger_type;

BOOST_DECLARE_LOG_FILTER(g_log_filter, filter::no_ts ) 
BOOST_DECLARE_LOG(g_l, logger_type) 

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
    for ( int i = 0; i < 20; ++i) {
        L_ << "message " << i ;
        do_sleep(1);
    }
}

void ts_logger_one_filter_example() {
    g_l()->writer().add_formatter( formatter::idx(), "[%] "  );
    g_l()->writer().add_formatter( formatter::time("$mm:$ss ") );
    g_l()->writer().add_formatter( formatter::append_newline() );
    g_l()->writer().add_destination( destination::file("out.txt") );
    g_l()->writer().add_destination( destination::dbg_window() );
    g_l()->mark_as_initialized();

    for ( int i = 0 ; i < 5; ++i)
        boost::thread t( &use_log_thread);

    // allow for all threads to finish
    std::cout << "sleep 5s " << std::endl;
    do_sleep( 5000);
}




int main() {
    ts_logger_one_filter_example();
}


// End of file


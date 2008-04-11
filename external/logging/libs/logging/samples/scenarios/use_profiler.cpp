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
@example use_profiler.cpp

@copydoc use_profiler

@page use_profiler use_profiler.cpp Example


This usage:

- You have one logger, and want to profile it:
  - you want to know how much time is spent while logging
  - Test: dump some dummy string 5000 times, and then the profiling information is written to "profile.txt"
- The logger has several log destinations
    - The output goes to console, debug output window, and a file called out.txt
    - Formatting - prefix each message by its index, and append newline

Optimizations:
- use a cache string (from optimize namespace), in order to make formatting the message faster

If logging on dedicated thread, the output for "profile.txt" could look like:

@code
gather time:      0.796875 seconds 
write time:       0.78125 seconds 
filter time:      0.15625 seconds 
otherthread time: 1.156250 seconds 
@endcode



\n\n

If logging on same thread, the output for "profile.txt" could look like:

@code
gather time:      5.562500 seconds 
write time:       5.265625 seconds 
filter time:      0.31250 seconds 
otherthread time: 0.0 seconds 
@endcode


*/

// if this is defined, we do the logging on a dedicated thread
// otherwise, logging happens in the thread that does the logging
//
// comment/uncomment this and play with this sample, to see the differences...
#define PROFILE_ON_DEDICATED_THREAD


#include <boost/logging/format_fwd.hpp>

BOOST_LOG_FORMAT_MSG( optimize::cache_string_one_str<> )

#include <boost/logging/format_ts.hpp>
#include <boost/logging/profile.hpp>

namespace bl = boost::logging;

////////////////////////////////////////////////////////////////////////////////////
// Profiling code

#if defined(PROFILE_ON_DEDICATED_THREAD)
typedef bl::logger_format_write< bl::default_, bl::default_, bl::writer::threading::on_dedicated_thread > raw_log_type;
#else
typedef bl::logger_format_write< > raw_log_type;
#endif
typedef bl::profile::compute_for_logger<raw_log_type>::type logger_type;

typedef bl::profile::compute_for_filter<bl::filter::no_ts>::type filter_type;

// END OF Profiling code
////////////////////////////////////////////////////////////////////////////////////




#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) 

BOOST_DEFINE_LOG_FILTER(g_log_filter, filter_type ) 
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


void one_logger_one_filter_example() {
    g_l()->writer().add_formatter(      bl::formatter::idx(), "[%] "  );
    g_l()->writer().add_formatter(      bl::formatter::append_newline() );
    g_l()->writer().add_destination(    bl::destination::file("out.txt") );
    g_l()->writer().add_destination(    bl::destination::cout() );
    g_l()->writer().add_destination(    bl::destination::dbg_window() );
    g_l()->mark_as_initialized();

    // where shall the profile results be outputted?
    bl::profile::compute::inst().log_results( bl::destination::file("profile.txt") );

    for ( int i = 0; i < 5000; ++i)
        L_ << "this is so cool " << i;

#if defined(PROFILE_ON_DEDICATED_THREAD)
    std::cout << "waiting for logging to finish" << std::endl;
    // wait for the logging to take place
    do_sleep(1000);
#endif
}




int main() {
    one_logger_one_filter_example();
}


// End of file


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
@example using_tags.cpp

@copydoc using_tags

@page using_tags using_tags.cpp Example


This usage:
- You have one logger and one filter, which can be turned on or off
- You want to format the message before it's written 
- The logger has several log destinations
    - The output goes to console, and a file called out.txt
    - Formatting - message will look like this: <tt>[file/line] [thread_id] [idx] [time] message [enter] </tt>

Optimizations:
- use a cache string (from optimize namespace), in order to make formatting the message faster

In this example, all output will be written to the console, debug window, and "out.txt" file.
The output can look like:

@code
logging\samples\scenarios\using_tags.cpp:94 [T7204] [1] 14:55 this is so cool 1
logging\samples\scenarios\using_tags.cpp:95 [T7204] [2] 14:55 this is so cool again 2
@endcode

*/



#include <boost/logging/format_fwd.hpp>

namespace bl = boost::logging;
typedef bl::tag::holder< bl::optimize::cache_string_one_str<>, bl::tag::file_line, bl::tag::thread_id, bl::tag::time> log_string;
BOOST_LOG_FORMAT_MSG( log_string )


#include <boost/logging/format_ts.hpp>
#include <boost/logging/format/formatter/tags.hpp>
#include <boost/logging/format/formatter/named_spacer.hpp>

using namespace boost::logging;

using namespace boost::logging::scenario::usage;
typedef use<
        //  the filter is always accurate (but slow)
        filter_::change::always_accurate, 
        //  filter does not use levels
        filter_::level::no_levels, 
        // the logger is initialized once, when only one thread is running
        logger_::change::set_once_when_one_thread, 
        // the logger favors speed (on a dedicated thread)
        logger_::favor::speed> finder;

BOOST_DECLARE_LOG_FILTER(g_log_filter, finder::filter ) 
BOOST_DECLARE_LOG(g_l, finder::logger) 

#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) .set_tag( BOOST_LOG_TAG_FILELINE)

BOOST_DEFINE_LOG_FILTER(g_log_filter, finder::filter ) 
BOOST_DEFINE_LOG(g_l, finder::logger) 


void using_tags_example() {
    //         add formatters and destinations
    //         That is, how the message is to be formatted and where should it be written to

    g_l()->writer().add_formatter( formatter::named_spacer( "%fileline% [T%thread_id%] [%idx%] %time%" )
        .add( "time", formatter::tag::time("$mm:$ss ") )                // time tag
        .add( "idx", formatter::idx() )                            
        .add( "thread_id", formatter::tag::thread_id() )                // thread_id tag
        .add( "fileline", formatter::tag::file_line() ) );              // file/line tag

    g_l()->writer().add_formatter( formatter::append_newline() );     
    g_l()->writer().add_destination( destination::cout() );
    g_l()->writer().add_destination( destination::file("out.txt") );
    g_l()->mark_as_initialized();

    int i = 1;
    L_ << "this is so cool " << i++;
    L_ << "this is so cool again " << i++;
}




int main() {
    using_tags_example();
}


// End of file


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
@example use_tss_ostringstream.cpp

@copydoc use_tss_ostringstream

@page use_tss_ostringstream use_tss_ostringstream.cpp Example


This usage:
- You have one logger
- You have one filter, always turned on
- You want to format the message before it's written 
- The logger has several log destinations
    - The output goes to console and debug output window
    - Formatting - prefix each message by its index, and append newline

Optimizations:
- use tss_ostringstream (each thread has its own ostringstream copy, to make writing faster: 
  when logging of a message, we won't need to create the ostringstream first ; it's created only once per thread )
- use a cache string (from optimize namespace), in order to make formatting the message faster


In this example, all output will be written to the console and debug window.
It will be:

@code
[1] this is so cool 1
[2] this is so cool again 2
[3] this is too cool 3
@endcode

*/



#include <boost/logging/format_fwd.hpp>

BOOST_LOG_FORMAT_MSG( optimize::cache_string_one_str<> )

// FIXME need to set the gather class

#include <boost/logging/format.hpp>

using namespace boost::logging;

typedef logger_format_write< > logger_type;

BOOST_DECLARE_LOG_FILTER(g_log_filter, filter::no_ts ) 
BOOST_DECLARE_LOG(g_l, logger_type) 

#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) 

BOOST_DEFINE_LOG_FILTER(g_log_filter, filter::no_ts ) 
BOOST_DEFINE_LOG(g_l, logger_type)


void use_tss_ostringstream_example() {
    //         add formatters and destinations
    //         That is, how the message is to be formatted and where should it be written to

    g_l()->writer().add_formatter( formatter::idx(), "[%] "  );
    g_l()->writer().add_formatter( formatter::append_newline_if_needed() );
    g_l()->writer().add_destination( destination::cout() );
    g_l()->writer().add_destination( destination::dbg_window() );
    g_l()->mark_as_initialized();

    int i = 1;
    L_ << "this is so cool " << i++;
    L_ << "this is so cool again " << i++;
    L_ << "this is so too cool " << i++;
}




int main() {
    use_tss_ostringstream_example();
}


// End of file


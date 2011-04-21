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


// test_log_output.cpp
//
// Tests that logging messages are output correctly

#include <boost/test/minimal.hpp>

#define BOOST_LOG_COMPILE_FAST_OFF
#include <boost/logging/format_fwd.hpp>

using namespace boost::logging;

BOOST_LOG_FORMAT_MSG( optimize::cache_string_one_str<> )

typedef logger_format_write< > log_type;

#include <boost/logging/format.hpp>


// Step 4: declare which filters and loggers you'll use (usually in a header file)
BOOST_DECLARE_LOG_FILTER(g_log_filter, filter::no_ts ) 
BOOST_DECLARE_LOG(g_l, log_type) 
BOOST_DEFINE_LOG_FILTER(g_log_filter, filter::no_ts ) 
BOOST_DEFINE_LOG(g_l, log_type) 

#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) 

void test_log_output() {
    std::ostringstream out_str;
    destination::stream dest_out(out_str);
    g_l()->writer().add_formatter( formatter::idx(), "[%] " );
    g_l()->writer().add_formatter( formatter::append_newline() );
    g_l()->writer().add_destination( destination::cout() );
    g_l()->writer().add_destination( dest_out );
    g_l()->mark_as_initialized();

    // Step 8: use it...
    int i = 1;
    L_ << "this is so cool " << i++;
    L_ << "this is so cool again " << i++;

    // does not output to our stringstream
    dest_out.clear();
    L_ << "only to console " << i++;

    std::string logged_msg = out_str.str();
    BOOST_CHECK( logged_msg == "[1] this is so cool 1\n[2] this is so cool again 2\n");
}



int test_main(int, char *[]) { 
    test_log_output() ; 
    return 0;
}


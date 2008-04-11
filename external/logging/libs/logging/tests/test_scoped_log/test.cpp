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

/* 
    Tests scoped_log
*/

#include <boost/test/minimal.hpp>

#include <boost/logging/format.hpp>
#include <sstream>

using namespace boost::logging;

typedef logger_format_write< > log_type;

BOOST_DEFINE_LOG_FILTER(g_log_filter, filter::no_ts ) 
BOOST_DEFINE_LOG(g_l, log_type)

#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) 

void scoped_func(int a, std::string str) {
    BOOST_SCOPED_LOG_CTX(L_) << "func(" << a << ", str=" << str << ")";
    L_ << "inner";
}

std::ostringstream g_out;
int test_main(int, char *[]) { 
    g_l()->writer().add_formatter( formatter::idx(), "[%] ");
    g_l()->writer().add_formatter( formatter::append_newline() );
    g_l()->writer().add_destination( destination::stream(g_out) );
    g_l()->mark_as_initialized();

    scoped_func(1, "str");
    g_log_filter()->set_enabled(false);
    scoped_func(2, "str2");
    g_log_filter()->set_enabled(true);
    scoped_func(3, "str3");

    std::string out = g_out.str();
    BOOST_CHECK( out == 
        "[1] start of func(1, str=str)\n"
        "[2] inner\n"
        "[3]   end of func(1, str=str)\n"
        "[4] start of func(3, str=str3)\n"
        "[5] inner\n"
        "[6]   end of func(3, str=str3)\n"
        );
    return 0;
}

// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details




#include <boost/logging/format_fwd.hpp>

#include <boost/logging/format/named_write.hpp>
typedef boost::logging::named_logger<>::type logger_type;

#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) 

// Define the filters and loggers you'll use (usually in a source file)
BOOST_DEFINE_LOG_FILTER(g_log_filter, boost::logging::filter::no_ts ) 
BOOST_DEFINE_LOG(g_l, logger_type)


void one_logger_one_filter_example() {
    // formatting    : [idx] message \n
    // destinations  : console, file "out.txt" and debug window
    g_l()->writer().write("[%idx%] |\n", "cout file(out.txt) debug");
    g_l()->mark_as_initialized();

    int i = 1;
    L_ << "this is so cool " << i++;
    L_ << "this is so cool again " << i++;

    std::string hello = "hello", world = "world";
    L_ << hello << ", " << world;

    g_log_filter()->set_enabled(false);
    L_ << "this will not be written to the log";
    L_ << "this won't be written to the log";

    g_log_filter()->set_enabled(true);
    L_ << "good to be back ;) " << i++;
}




int main() {
    one_logger_one_filter_example();
}


// End of file


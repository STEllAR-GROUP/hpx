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
    Tests formatter::file
*/
#include <boost/test/minimal.hpp>

#include <boost/logging/format.hpp>
#include <string>

using namespace boost::logging;

typedef logger_format_write< > log_type;

BOOST_DEFINE_LOG_FILTER(g_log_filter, filter::no_ts ) 
BOOST_DEFINE_LOG(g_l, log_type)

#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) 


void write_to_clean_file() {
    // first, write to a clean file (that is, don't append to it)
    g_l()->writer().add_destination( destination::file("out.txt", destination::file_settings().initial_overwrite(true) ));
    g_l()->writer().add_formatter( formatter::append_newline_if_needed() );
    g_l()->writer().add_destination( destination::cout() );
    g_l()->turn_cache_off();

    // read this .cpp file - every other line is logged (odd lines)
    std::ifstream in("test.cpp");
    bool enabled = true;
    std::string line;
    while ( std::getline(in, line) ) {
        g_log_filter()->set_enabled(enabled);
        L_ << "line odd " << line;
        enabled = !enabled;
    }
}

void append_to_file() {
    // second, append to the same file

    // ... first, remove old destination
    g_l()->writer().del_destination( destination::file("out.txt"));
    // ... now, re-add the same file - but now, for appending
    g_l()->writer().add_destination( destination::file("out.txt", 
        destination::file_settings().initial_overwrite(false).do_append(true) ));

    // read this .cpp file - every other line is logged (even lines now)
    std::ifstream in("test.cpp");
    bool enabled = false;
    std::string line;
    while ( std::getline(in, line) ) {
        g_log_filter()->set_enabled(enabled);
        L_ << "line even " << line;
        enabled = !enabled;
    }

    g_l()->writer().del_destination( destination::file("out.txt"));
    g_log_filter()->set_enabled(true);
    L_ << "should not be written to file, only to console";
}

// now, see that what we've written was ok
void test_write_ok() {
    std::ifstream test("test.cpp");
    std::ifstream out("out.txt");
    std::string test_line, out_line;
    // first, odd lines
    while ( std::getline(test, test_line) ) {
        std::getline(out, out_line);
        BOOST_CHECK( "line odd " + test_line == out_line );
        std::getline(test, test_line); // ignore even line
    }

    test.close();
    std::ifstream test2("test.cpp");

    // second, even lines
    while ( std::getline(test2, test_line) && std::getline(test2, test_line) ) {
        std::getline(out, out_line);
        BOOST_CHECK( "line even " + test_line == out_line );
    }

    // out.txt - should have no more lines
    std::getline(out, out_line);
    BOOST_CHECK( out_line.empty() );
}


int test_main(int, char *[]) { 
    write_to_clean_file();
    append_to_file();
    test_write_ok();
    return 0;
}


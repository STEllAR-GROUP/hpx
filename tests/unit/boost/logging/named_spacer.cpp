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
    Tests named_spacer
*/

#include <boost/test/minimal.hpp>

#include <boost/logging/format.hpp>
#include <boost/logging/format/formatter/named_spacer.hpp>

using namespace boost::logging::scenario::usage;
typedef use< filter_::change::single_thread, filter_::level::no_levels, logger_::change::single_thread, logger_::favor::single_thread > finder;

using namespace boost::logging;

BOOST_DEFINE_LOG_FILTER(g_log_filter, finder::filter  ) 
BOOST_DEFINE_LOG(g_l, finder::logger )

#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) 


// whatever we log, is logged here too (easy was to find out all the info that was logged)
std::stringstream g_out;


// writes a letter (a-z); on each write, it increments the letter, and rolls back to 'a' once reaching 'z'
struct abc : formatter::class_<abc, formatter::implement_op_equal::no_context> {
    abc() : cur_letter('a') {}

    void operator()(std::string & str) const {
        str = cur_letter + str;
        if ( ++cur_letter > 'z')
            cur_letter = 'a';
    }

    mutable char cur_letter;
};

// our named spacer - the one we're testing
formatter::named_spacer_t<boost::logging::default_, boost::logging::default_, lock_resource_finder::single_thread > g_ns;

// we're constantly writing hello world
std::string g_msg = "hello world";

// current thread id - note : in our processing, we only need to know it as string.
std::string g_thread_id;

void init_logs() {
    g_l()->writer().add_formatter( g_ns
        .add( "idx", formatter::idx() )
        .add( "tid", formatter::thread_id() )
        .add( "abc", abc() ));
    g_l()->writer().add_formatter( formatter::append_newline() );
    g_l()->writer().add_destination( destination::stream(g_out) );
    g_l()->writer().add_destination( destination::cout() );
    g_l()->turn_cache_off();
}

void test_with_all_formatters() {
    g_ns.string("[%idx%] {%tid%} (%abc%)-");
    L_ << g_msg;
    BOOST_CHECK( g_out.str() == "[1] {" + g_thread_id + "} (a)-hello world\n");
    g_out.str("");

    g_ns.string("[%idx%] (%abc%)-{%tid%}/");
    L_ << g_msg;
    BOOST_CHECK( g_out.str() == "[2] (b)-{" + g_thread_id + "}/hello world\n");
    g_out.str("");

    g_ns.string("[%idx%]/[%abc%]/[%tid%]/ ");
    L_ << g_msg;
    BOOST_CHECK( g_out.str() == "[3]/[c]/[" + g_thread_id + "]/ hello world\n");
    g_out.str("");
}

void test_with_2_formatters() {
    g_ns.string("[%idx%] (%abc%)-");
    L_ << g_msg;
    BOOST_CHECK( g_out.str() == "[4] (d)-hello world\n");
    g_out.str("");

    g_ns.string("[%tid%] (%idx%)-");
    L_ << g_msg;
    BOOST_CHECK( g_out.str() == "[" + g_thread_id + "] (5)-hello world\n");
    g_out.str("");

    g_ns.string("[%abc%] [%tid%]: ");
    L_ << g_msg;
    BOOST_CHECK( g_out.str() == "[e] [" + g_thread_id + "]: hello world\n");
    g_out.str("");

}

void test_with_1_formatter() {
    g_ns.string("[%idx%]/ ");
    L_ << g_msg;
    BOOST_CHECK( g_out.str() == "[6]/ hello world\n");
    g_out.str("");

    g_ns.string("{%tid%}- ");
    L_ << g_msg;
    BOOST_CHECK( g_out.str() == "{" + g_thread_id + "}- hello world\n");
    g_out.str("");

    g_ns.string("%abc%/ ");
    L_ << g_msg;
    BOOST_CHECK( g_out.str() == "f/ hello world\n");
    g_out.str("");

}


void test_with_no_formatters() {
    g_ns.string("/ ");
    L_ << g_msg;
    BOOST_CHECK( g_out.str() == "/ hello world\n");
    g_out.str("");

    g_ns.string("abc ");
    L_ << g_msg;
    BOOST_CHECK( g_out.str() == "abc hello world\n");
    g_out.str("");

    g_ns.string("");
    L_ << g_msg;
    BOOST_CHECK( g_out.str() == "hello world\n");
    g_out.str("");

}


int test_main(int, char *[]) { 
    init_logs();
    std::ostringstream out;
    out << detail::get_thread_id();
    g_thread_id = out.str();

    test_with_all_formatters();
    test_with_2_formatters();
    test_with_1_formatter();
    test_with_no_formatters();
    return 0;
}

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
    Tests using tags
    - tests writing file/line & function
    - tests dumping the module and the thread_id
    
*/
#include <boost/test/minimal.hpp>

#include <boost/logging/format_fwd.hpp>

// Optimize : use tags 
namespace bl = boost::logging;
typedef bl::tag::holder< bl::default_, bl::tag::file_line, bl::tag::function, bl::tag::thread_id, bl::tag::module> log_string;
BOOST_LOG_FORMAT_MSG( log_string )

#include <boost/logging/format.hpp>
#include <boost/logging/format/formatter/tags.hpp>

using namespace boost::logging::scenario::usage;
typedef use< filter_::change::single_thread, filter_::level::no_levels, logger_::change::single_thread, logger_::favor::single_thread > finder;

using namespace boost::logging;

BOOST_DEFINE_LOG_FILTER(g_log_filter, finder::filter  ) 
BOOST_DEFINE_LOG(g_l, finder::logger )





// this macro is used to write the file/line and function
#define LOG_FILE_FUNC_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) .set_tag( BOOST_LOG_TAG_FILELINE) .set_tag( BOOST_LOG_TAG_FUNCTION)

// this macro is used to write module and thread id
#define LOG_MODULE_(mod) BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) .set_tag( BOOST_LOG_TAG(module)(mod) ) 


// whatever we log, is logged here too (easy was to find out all the info that was logged)
std::stringstream g_out;

// we're constantly writing hello world
std::string g_msg = "hello world";

void init_logs_fileline_function() {
    g_l()->writer().add_formatter( formatter::tag::file_line()); //, "[%] " );     
    g_l()->writer().add_formatter( formatter::tag::function()); //, "% " );     
    g_l()->writer().add_formatter( formatter::append_newline() );     
    g_l()->writer().add_destination( destination::stream(g_out) );
    g_l()->writer().add_destination( destination::cout() );
    g_l()->turn_cache_off();
}

// the str should contain 
bool test_str_contains(const std::string & str, const std::string & find) {
    return str.find(find) != std::string::npos;
}

void test_fileline_function() {
    init_logs_fileline_function();
    LOG_FILE_FUNC_ << g_msg;
    BOOST_CHECK( test_str_contains( g_out.str(), "test_tags.cpp"));
    BOOST_CHECK( test_str_contains( g_out.str(), "test_fileline_function"));
    g_out.str("");

    LOG_FILE_FUNC_ << g_msg;
    BOOST_CHECK( test_str_contains( g_out.str(), "test_tags.cpp"));
    BOOST_CHECK( test_str_contains( g_out.str(), "test_fileline_function"));
    g_out.str("");
}


void init_logs_module_thread_id() {
    g_l()->writer().del_formatter( formatter::tag::file_line() );     
    g_l()->writer().del_formatter( formatter::tag::function() );     

    // after deleting the file/line and function formatters, they shouldn't be called
    LOG_FILE_FUNC_ << g_msg;
    BOOST_CHECK( !test_str_contains( g_out.str(), "test_tags.cpp"));
    BOOST_CHECK( !test_str_contains( g_out.str(), "test_fileline_function"));
    g_out.str("");

    g_l()->writer().add_formatter( formatter::tag::module(), "[%] " );     
    g_l()->writer().add_formatter( formatter::tag::thread_id(), "[%] " );     
}

void test_module_and_thread_id() {
    init_logs_module_thread_id();

    std::ostringstream out;
    out << detail::get_thread_id();
    std::string thread_id = out.str();

    LOG_MODULE_("module_a") << g_msg;
    BOOST_CHECK( test_str_contains( g_out.str(), "[module_a]"));
    BOOST_CHECK( test_str_contains( g_out.str(), "[" + thread_id + "]"));
}

int test_main(int, char *[]) { 
    test_fileline_function();
    test_module_and_thread_id();
    return 0;
}


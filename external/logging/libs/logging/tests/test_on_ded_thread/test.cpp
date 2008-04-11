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
    Tests on_dedicated_thread
*/


#include <boost/test/minimal.hpp>


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

int MESSAGES_PER_THREAD = 200;
int THREAD_COUNT = 5;
void use_log_thread() {
    for ( int i = 0; i < MESSAGES_PER_THREAD; ++i) {
        L_ << detail::get_thread_id() << " message " << (i+1) ;
        do_sleep(1);
    }
}

std::ostringstream g_out;

void test_logged_messages() {
    typedef std::map<detail::thread_id_type, int> coll;
    coll messages;
    std::istringstream in( g_out.str());
    std::string line;
    while ( std::getline( in, line ) ) {
        // each logged message has this syntax:
        // thread_id message idx
        detail::thread_id_type thread_id;
        std::string word;
        int idx = 0;
        std::istringstream line_in(line);
        line_in >> thread_id >> word >> idx;
        BOOST_CHECK(word == "message");

        // for each thread - we should be reading an increasing index 1, 2, 3, ...
        BOOST_CHECK( messages[thread_id] + 1 == idx);
        messages[thread_id] = idx;
        // we only have THREAD_COUNT threads logging!
        BOOST_CHECK( (int)messages.size() <= THREAD_COUNT);
    }

    // at this point, each thread should have written all MESSAGES_PER_THREAD messages
    BOOST_CHECK( (int)messages.size() == THREAD_COUNT);
    for ( coll::const_iterator b = messages.begin(), e = messages.end() ; b != e; ++b)
        BOOST_CHECK( b->second == MESSAGES_PER_THREAD);
}


void test_on_dedicated_thread() {
    g_l()->writer().add_formatter( formatter::append_newline() );
    g_l()->writer().add_destination( destination::stream(g_out) );
    g_l()->writer().add_destination( destination::dbg_window() );
    g_l()->mark_as_initialized();

    for ( int i = 0 ; i < THREAD_COUNT; ++i)
        boost::thread t( &use_log_thread);

    // allow for all threads to finish
    int sleep_ms = MESSAGES_PER_THREAD * THREAD_COUNT * 2 ;
    std::cout << "sleeping for " << sleep_ms << " milliseconds" << std::endl;
    do_sleep( MESSAGES_PER_THREAD * THREAD_COUNT * 2 /* just in case*/);
    test_logged_messages();
}




int test_main(int, char *[]) { 
    test_on_dedicated_thread();
    return 0;
}


// End of file


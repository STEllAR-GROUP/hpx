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
    Test : we use TSS (Thread Specific Storage). 
    
    We have a dummy object that uses TSS.
    We check to see that every time I request the pointer to an object, on a given thread,
    the same object is returned.

*/
#include <boost/test/minimal.hpp>

#define BOOST_LOG_TSS_USE_INTERNAL
// this includes tss_value class
#include <boost/logging/logging.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/xtime.hpp>
#include <sstream>
#include <string>
#include <iostream>

struct object_count {
    typedef boost::mutex mutex;
    typedef mutex::scoped_lock scoped_lock;

    object_count() : m_count(0) {
    }

    ~object_count() {
        scoped_lock lk(m_cs);
    }

    int increment() {
        scoped_lock lk(m_cs);
        return ++m_count;
    }

    void decrement() {
        scoped_lock lk(m_cs);
        --m_count;
        BOOST_CHECK(m_count >= 0);
    }

    int count() const { 
        scoped_lock lk(m_cs);
        return m_count; 
    }

private:
    mutable mutex m_cs;
    int m_count;
};


struct dummy {
    std::string str;
};


using namespace boost;
using namespace logging;

void do_sleep(int ms) {
    xtime next;
    xtime_get( &next, TIME_UTC);
    next.nsec += (ms % 1000) * 1000000;

    int nano_per_sec = 1000000000;
    next.sec += next.nsec / nano_per_sec;
    next.sec += ms / 1000;
    next.nsec %= nano_per_sec;
    thread::sleep( next);
}


object_count g_running_threads_count ;

tss_value<dummy> g_dummy;

void use_dummy_thread() {
    int thread_idx = g_running_threads_count.increment();
    std::ostringstream out;
    out << thread_idx;
    // note: we want to create a unique string to each thread, so see that
    //       when manipulating local_str, we're actually manipulating the current thread's dummy object
    std::string thread_idx_str = out.str();

    dummy * local_dummy = &*g_dummy;
    std::string & local_str = g_dummy->str;
    std::string copy_str;
    
    // just in case we get an assertion fails - know when
    int try_idx = 0;

    while ( true) {
        ++try_idx;
        do_sleep(10);

        dummy * cur_dummy = &*g_dummy;
        if ( cur_dummy != local_dummy) {
            std::cout << "thread " << thread_idx << ": assertion failed - dummy - at try " << try_idx;
            BOOST_CHECK( false);
        }

        local_str += thread_idx_str;
        copy_str += thread_idx_str;
        if ( copy_str != g_dummy->str) {
            std::cout << "thread " << thread_idx << ": assertion failed - local_str - at try " << try_idx;
            BOOST_CHECK( false);
        }
    }

}




int g_total_thread_count = 20;

int g_run_test_secs = 10;

int test_main(int, char *[]) { 
    for ( int i = 0; i < g_total_thread_count ; ++i)
        thread t( &use_dummy_thread);

    std::cout << "running test for " << g_run_test_secs << " secs " << std::endl;
    do_sleep( g_run_test_secs * 1000 );
    std::cout << "done " << std::endl;
	return 0;
}


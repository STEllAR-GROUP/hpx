// test_tss.cpp : Defines the entry point for the console application.
//

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

/* 
    Test : we use TSS (Thread Specific Storage). We check to see that there are no objects leaked.

    We create a number of threads.
    Each thread :
    - reads the current file - word by word
    - after each read, pauses a bit (like, 1 ms)
    - the words are concatenated, ignoring spaces
    - after the whole file is read, dumps to console : the thread index and the character at the thread's index from the concatenated string
    - we read as many words as there are threads (in order not to take too much time)

    The thread information is all kept using TSS



    Note: I've split the logic on multiple files, to test to see that we don't run into the Meyers Singleton bug.

*/

// so that we can catch the end of deleting all objects
#define BOOST_LOG_TEST_TSS

#define BOOST_LOG_TSS_USE_INTERNAL
// this includes tss_value class
#include <boost/logging/logging.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>
#include <boost/bind.hpp>
#include <boost/assert.hpp>
#include <boost/shared_ptr.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "count.h"

using namespace boost;

// creating this (a log, or a filter), makes sure we initialize TSS
BOOST_DEFINE_LOG_FILTER_WITH_ARGS(g_log_filter, logging::filter::use_tss_with_cache<> , 10) 

extern object_count * g_object_count ;
extern object_count * g_running_thread_count ;

// the actual number of started threads
int g_thread_count = 50;

void do_sleep(int ms) ; 
void process_file() ;

int test_main(int argc, char *argv[]) { 
    if ( argc > 1) {
        std::istringstream in(argv[1]);
        in >> g_thread_count;
    }
    std::cout << "running test with " << g_thread_count << " threads" << std::endl;

    for ( int idx = 0; idx < g_thread_count ; ++idx)
        thread t( &process_file);

    do_sleep(1000);
    while ( g_running_thread_count->count() > 0 ) {
        do_sleep(100);
        std::cout << "remaining running threads " << g_running_thread_count->count() << std::endl;
    }
	return 0;
}



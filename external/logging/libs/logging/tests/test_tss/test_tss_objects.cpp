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

// so that we can catch the end of deleting all objects
#define BOOST_LOG_TEST_TSS
#include <boost/test/minimal.hpp>

#define BOOST_LOG_TSS_USE_INTERNAL
// this includes tss_value class
#include <boost/logging/logging.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
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





// we specifically let these 2 objects leak - making them static, would destroy them too soon (before the other objects being destroyed)
object_count * g_object_count = new object_count;
object_count * g_running_thread_count = new object_count;

extern int g_thread_count ;



struct managed_object {
    managed_object(object_count & counter) : m_counter(counter) {
        m_counter.increment();
    }
    managed_object(const managed_object& other) : m_counter(other.m_counter) {
        m_counter.increment();
    }
    ~managed_object() {
        m_counter.decrement();
    }
private:
    object_count & m_counter;
};

struct read_file : private managed_object {
    // read this file itself
    read_file() : managed_object(*g_object_count), m_in(new std::ifstream("test_tss.cpp")), m_word_idx(0) {
    }

    std::string read_word() {
        ++m_word_idx;
        if ( m_word_idx <= g_thread_count) {
            std::string word;
            (*m_in) >> word;
            return word;
        }
        else
            return "";
    }

private:
    boost::shared_ptr<std::ifstream> m_in;
    int m_word_idx;
};

struct string_no_spaces : private managed_object {
    string_no_spaces() : managed_object(*g_object_count) {}

    void add_word(const std::string & word) {
        m_str += word;
    }

    char at_idx(int idx) {
        if ( idx < (int)m_str.size())
            return m_str[idx];
        else
            return ' ';
    }

private:
    std::string m_str;
};

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

using namespace logging;
tss_value<read_file> file;
tss_value<string_no_spaces> str;

void process_file() {
    managed_object m(*g_running_thread_count);
    int thread_idx = g_running_thread_count->count();
    std::cout << "started thread " << thread_idx << std::endl;

    read_file local_file;
    while ( true) {

        read_file * file_ptr = &*file;
        std::string word = file_ptr->read_word();
        std::string local_word = local_file.read_word();
        // it should behave just like a "local" variable
        if ( word != local_word)
            BOOST_CHECK( false);
        str->add_word(word);
        if ( word.empty() )
            break;
        do_sleep(5);
    }

    std::cout << "char at idx " << thread_idx << ":" << str->at_idx(thread_idx) << std::endl ;

    std::cout << "ended thread " << thread_idx << std::endl;
}


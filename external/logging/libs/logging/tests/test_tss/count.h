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

#ifndef object_count_h
#define object_count_h

#include <boost/thread/mutex.hpp>

/* 
    counts the number of objects.
    When it's destroyed, there should be no objects left.
*/
struct object_count {
    typedef boost::mutex mutex;
    typedef mutex::scoped_lock scoped_lock;

    object_count() : m_count(0) {
    }

    ~object_count() {
        scoped_lock lk(m_cs);
        BOOST_ASSERT(m_count == 0);
    }

    void increment() {
        scoped_lock lk(m_cs);
        ++m_count;
    }

    void decrement() {
        scoped_lock lk(m_cs);
        --m_count;
        BOOST_ASSERT(m_count >= 0);
    }

    int count() const { 
        scoped_lock lk(m_cs);
        return m_count; 
    }

private:
    mutable mutex m_cs;
    int m_count;
};

#endif

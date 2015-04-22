// ts_posix.hpp

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



// Copyright (C) 2001-2003
// William E. Kempf
//
// Permission to use, copy, modify, distribute and sell this software
// and its documentation for any purpose is hereby granted without fee,
// provided that the above copyright notice appear in all copies and
// that both that copyright notice and this permission notice appear
// in supporting documentation.  William E. Kempf makes no representations
// about the suitability of this software for any purpose.
// It is provided "as is" without express or implied warranty.


#ifndef JT28092007_HPX_LOG_TS_HPP_posix
#define JT28092007_HPX_LOG_TS_HPP_posix


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#if !defined(HPX_HAVE_LOG_NO_TS)

#include <errno.h>
#include <pthread.h>
#include <stdexcept>
#include <hpx/util/assert.hpp>

namespace hpx { namespace util { namespace logging {

namespace threading {

class scoped_lock_posix ;

class mutex_posix {

    mutex_posix & operator = ( const mutex_posix & Not_Implemented);
    mutex_posix( const mutex_posix & From);
public:
    typedef scoped_lock_posix scoped_lock;

    mutex_posix() : m_mutex(), m_count(0) {
        pthread_mutexattr_t attr;
        int res = pthread_mutexattr_init(&attr);
        HPX_ASSERT(res == 0);

        res = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
        HPX_ASSERT(res == 0);

        res = pthread_mutex_init(&m_mutex, &attr);
        {
            int r = 0;
            r = pthread_mutexattr_destroy(&attr);
            HPX_ASSERT(r == 0);
        }
        if (res != 0)
            throw std::runtime_error("could not create mutex_posix");
    }
    ~mutex_posix() {
        int res = 0;
        res = pthread_mutex_destroy(&m_mutex);
        HPX_ASSERT(res == 0);
    }

    void Lock() {
        int res = 0;
        res = pthread_mutex_lock(&m_mutex);
        HPX_ASSERT(res == 0);
        if (++m_count > 1)
        {
            res = pthread_mutex_unlock(&m_mutex);
            HPX_ASSERT(res == 0);
        }
    }
    void Unlock() {
        if (--m_count == 0)
        {
            int res = 0;
            res = pthread_mutex_unlock(&m_mutex);
            HPX_ASSERT(res == 0);
        }
    }
private:
    pthread_mutex_t m_mutex;
    unsigned m_count;
};

class scoped_lock_posix {
    scoped_lock_posix operator=( scoped_lock_posix & Not_Implemented);
    scoped_lock_posix( const scoped_lock_posix & Not_Implemented);
public:
    scoped_lock_posix( mutex_posix & cs) : m_cs( cs)                { m_cs.Lock(); }
    ~scoped_lock_posix()                                      { m_cs.Unlock(); }
private:
    mutex_posix & m_cs;
};

}

}}}

#endif

#endif


// ts_win32.hpp

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


#ifndef JT28092007_HPX_LOG_TS_HPP_win32
#define JT28092007_HPX_LOG_TS_HPP_win32


#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#if !defined(HPX_HAVE_LOG_NO_TS)

// many thanks to Terris Linerbach!
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>


namespace hpx { namespace util { namespace logging { namespace threading {

class scoped_lock_win32 ;

class mutex_win32 {

    mutex_win32 & operator = ( const mutex_win32 & Not_Implemented);
    mutex_win32( const mutex_win32 & From);
public:
    typedef scoped_lock_win32 scoped_lock;

    mutex_win32() {
        InitializeCriticalSection( GetCriticalSectionPtr() );
    }
    ~mutex_win32() {
        DeleteCriticalSection(GetCriticalSectionPtr());
    }
    void Lock() {
        EnterCriticalSection( GetCriticalSectionPtr());
    }
    void Unlock() {
        LeaveCriticalSection( GetCriticalSectionPtr());
    }
private:
    LPCRITICAL_SECTION GetCriticalSectionPtr() const { return &m_cs; }
    mutable CRITICAL_SECTION m_cs;
};

class scoped_lock_win32 {
    scoped_lock_win32 operator=( scoped_lock_win32 & Not_Implemented);
    scoped_lock_win32( const scoped_lock_win32 & Not_Implemented);
public:
    scoped_lock_win32( mutex_win32 & cs) : m_cs( cs)                { m_cs.Lock(); }
    ~scoped_lock_win32()                                      { m_cs.Unlock(); }
private:
    mutex_win32 & m_cs;
};

}}}}

#endif

#endif


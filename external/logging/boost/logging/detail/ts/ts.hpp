// ts.hpp

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

#ifndef JT28092007_BOOST_LOG_TS_HPP
#define JT28092007_BOOST_LOG_TS_HPP


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <boost/config.hpp>

#include <boost/logging/detail/ts/ts_none.hpp>

#ifdef BOOST_HAS_THREADS
    
    #ifdef BOOST_LOG_USE_BOOST_THREADS
        #include <boost/logging/detail/ts/ts_boost.hpp>
    #else
        #ifdef BOOST_WINDOWS
        #include <boost/logging/detail/ts/ts_win32.hpp>
        #else
        #include <boost/logging/detail/ts/ts_posix.hpp>
        #endif
    #endif

#else
    // no threads
    #include <boost/logging/detail/ts/ts_none.hpp>
#endif



namespace boost { namespace logging { namespace threading {

#ifdef BOOST_HAS_THREADS
    
    #ifdef BOOST_LOG_USE_BOOST_THREADS
        typedef mutex_boost mutex;
    #else
        #ifdef BOOST_WINDOWS
        typedef mutex_win32 mutex;
        #else
        typedef mutex_posix mutex;
        #endif
    #endif

#else
    // no threads
    typedef no_mutex mutex;
#endif


typedef mutex::scoped_lock scoped_lock;

}}}

#endif


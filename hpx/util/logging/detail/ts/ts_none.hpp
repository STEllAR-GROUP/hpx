// ts_none.hpp

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


#ifndef JT28092007_HPX_LOG_TS_HPP_none
#define JT28092007_HPX_LOG_TS_HPP_none

namespace hpx { namespace util { namespace logging {

namespace threading {
    // no threads
    struct no_mutex ;

    struct no_lock {
        no_lock(no_mutex &) {}
        ~no_lock() {}
    };

    struct no_mutex {
        typedef no_lock scoped_lock;
        void Lock() {}
        void Unlock() {}
    };
}

}}}

#endif


// ts_boost.hpp

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

#ifndef JT28092007_HPX_LOG_TS_HPP_boost
#define JT28092007_HPX_LOG_TS_HPP_boost

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#if !defined(HPX_HAVE_LOG_NO_TS)

#include <boost/thread/mutex.hpp>

namespace hpx { namespace util { namespace logging {

namespace threading {

    typedef boost::mutex mutex_boost;
    typedef mutex::scoped_lock scoped_lock_boost;
}

}}}

#endif

#endif


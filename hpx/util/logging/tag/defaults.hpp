// tag_defaults.hpp

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


#ifndef JT28092007_tag_defaults_HPP_DEFINED
#define JT28092007_tag_defaults_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>
#include <hpx/util/logging/logging.hpp>
//#include <boost/thread/detail/config.hpp>
#include <boost/config.hpp>

namespace hpx { namespace util { namespace logging { namespace tag {

/** @brief tag that holds file/line context information

See @ref hpx::util::logging::tag "how to use tags".
*/
struct file_line {
    file_line(const char * val_ = "") : val(val_) {}
    const char * val;
};

/** @brief tag that holds function name context information

See @ref hpx::util::logging::tag "how to use tags".
*/
struct function {
    function(const char* name = "") : val(name) {}
    const char * val;
};

/** @brief tag that holds the log level context information

See @ref hpx::util::logging::tag "how to use tags".
*/
struct level {
    level(::hpx::util::logging::level::type val_ = 0) : val(val_) {}
    ::hpx::util::logging::level::type val;
};

/** @brief tag that holds the current time context information

See @ref hpx::util::logging::tag "how to use tags".
*/
struct time {
    time() : val( ::time(0) ) {}
    ::time_t val;
};




/** @brief tag that holds module context information
(note: you need to specify the module yourself)

See @ref hpx::util::logging::tag "how to use tags".
*/
struct module {
    module(const char* name = "") : val(name) {}
    const char * val;
};


/** @brief tag that holds thread id context information

See @ref hpx::util::logging::tag "how to use tags".
*/
struct thread_id {
    thread_id() {
#if defined (BOOST_HAS_WINTHREADS)
        val = ::GetCurrentThreadId();
#elif defined (BOOST_HAS_PTHREADS)
        val = pthread_self ();
#else
#error Unknown type of threads
#endif
    }

#if defined (BOOST_HAS_WINTHREADS)
    DWORD val;
#elif defined (BOOST_HAS_PTHREADS)
    pthread_t val;
#endif
};

}}}}

#endif


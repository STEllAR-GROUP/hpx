// filter.hpp

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


#ifndef JT28092007_filter_HPP_DEFINED
#define JT28092007_filter_HPP_DEFINED

#include <hpx/util/logging/detail/fwd.hpp>

namespace hpx { namespace util {

/**
    @brief Root namespace. All the logging lib is contained in this namespace,
    or sub-namespaces of this one.
*/
namespace logging {


/**
    @brief Contains filter implementations. A filter tells the logger if
    it's enabled or not.


    The %filter namespace contains a few implementations of %filter classes.

    @c Filter is just a concept. You decide what a @c filter is.

    The minimalistic filter knows only if <i>it's enabled</i>

    Filter interface:
    @code
    struct some_filter class {
        // is filter enabled
        bool is_enabled() ;

        // ... whatever else you might want
    };
    @endcode

    In your logger, you can use any filter class that's already here,
    or implement your own.

    The filters defined by the library are:
    - filter::always_enabled
    - filter::always_disabled
    - filter::debug_enabled
    - filter::release_enabled
    - in case you use levels, see level namespace

*/
namespace filter {


/**
    @brief Manages is_enabled/set_enabled in a non-thread-safe way.

    If you change set_enabled() while program is running, it can take a bit to propagate
    between threads. Most of the time, this should be acceptable.
*/
struct no_ts {
    no_ts() : m_enabled(true) {}
    bool is_enabled() const { return m_enabled; }
    void set_enabled(bool enabled) { m_enabled = enabled; }
private:
    bool m_enabled;
};


/**
    @brief Filter that is always enabled
*/
struct always_enabled {
    static bool is_enabled() { return true; }
};


/**
    @brief Filter that is always disabled
*/
struct always_disabled {
    static bool is_enabled() { return false; }
};


/**
    @brief Filter that is enabled in debug mode
*/
struct debug_enabled {
#ifndef NDEBUG
    static bool is_enabled() { return true; }
#else
    static bool is_enabled() { return false; }
#endif
};


/**
    @brief Filter that is enabled in release mode
*/
struct release_enabled {
#ifdef NDEBUG
    static bool is_enabled() { return true; }
#else
    static bool is_enabled() { return false; }
#endif
};

} // namespace filter

}}}


#endif

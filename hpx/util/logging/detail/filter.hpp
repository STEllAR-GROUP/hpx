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

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>
#include <hpx/util/logging/detail/forward_constructor.hpp>
#include <hpx/util/logging/detail/tss/tss.hpp>

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
    or implement your own. Implementing a filter is usually as easy as it gets:

    @code
    struct filter_no_ts {
        filter_no_ts() : m_enabled(true) {}

        bool is_enabled() const { return m_enabled; }
        void set_enabled(bool enabled) { m_enabled = enabled; }
    private:
        bool m_enabled;
    };
    @endcode

    The filters defined by the library are:
    - filter::no_ts
    - filter::ts
    - filter::use_tss_with_cache
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


/**
    @brief Thread-safe filter. Manages is_enabled/set_enabled in a thread-safe way.

    However, it manages it rather ineffiently - always locking before asking.
*/
struct ts {
    ts() : m_enabled(true) {}
    bool is_enabled() const {
        threading::scoped_lock lk(m_cs);
        return m_enabled;
    }
    void set_enabled(bool enabled) {
        threading::scoped_lock lk(m_cs);
        m_enabled = enabled;
    }
private:
    bool m_enabled;
    mutable threading::mutex m_cs;
};




#ifndef HPX_HAVE_LOG_NO_TSS

/**
    @brief Uses TSS (Thread Specific Storage) to find out if a filter is enabled or not.

    It caches the current "is_enabled" on each thread.
    Then, at a given period, it retrieves the real "is_enabled".

    @remarks

    Another implementation can be done, which could be faster
    - where you retrieve the "is_enabled" each X calls on a given thread
    (like, every 20 calls on a given thread)
*/
template<int default_cache_secs = 5> struct use_tss_with_cache {
    typedef locker::tss_resource_with_cache<bool,default_cache_secs> data;

    use_tss_with_cache(int cache_secs = default_cache_secs)
        : m_enabled(true, cache_secs) {}
    bool is_enabled() const {
        typename data::read enabled(m_enabled);
        return enabled.use();
    }
    void set_enabled(bool enabled) {
        typename data::write cur(m_enabled);
        cur.use() = enabled;
    }
private:
    data m_enabled;
};

/**
    @brief Uses TSS (Thread Specific Storage) to find out if a filter is
    enabled or not. Once the filter is initialized to a value,
    that value will always be used.

*/
struct use_tss_once_init {
    typedef locker::tss_resource_once_init<bool> data;

    use_tss_once_init() : m_enabled(true) {}
    bool is_enabled() const {
        data::read enabled(m_enabled);
        return enabled.use();
    }
    void set_enabled(bool enabled) {
        data::write cur(m_enabled);
        cur.use() = enabled;
    }
private:
    data m_enabled;
};


#endif // #ifndef HPX_HAVE_LOG_NO_TSS


} // namespace filter

}}}


#endif


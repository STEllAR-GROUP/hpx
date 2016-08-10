// log_keeper.hpp

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


#ifndef JT28092007_log_keeper_HPP_DEFINED
#define JT28092007_log_keeper_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>
#include <hpx/util/logging/detail/forward_constructor.hpp>

#include <boost/cstdint.hpp>

namespace hpx { namespace util { namespace logging {

namespace detail {



} // namespace detail


/**
    @brief Allows using a log without knowing its full type yet.
    Even if the log is not fully @b defined, you can still use it.

    This will allow you to log messages even if you don't know the full type of the log
    (which can aid compilation time).

    This is a base class. Use logger_holder_by_value or logger_holder_by_ptr instead
*/
template<class type> struct logger_holder {
    typedef typename logger_to_gather<type> ::gather_type gather_type;
    typedef logger<gather_type> logger_base_type;

    const type* operator->() const      { return m_log; }
    type* operator->()                  { return m_log; }

    /**
        in case you want to get the real log object
    */
    const type* get() const             { return m_log; }
    type* get()                         { return m_log; }


    const logger_base_type * base() const    { return m_base; }
    logger_base_type * base()                { return m_base; }

protected:
    logger_holder() : m_log(nullptr), m_base(nullptr) {}
    virtual ~logger_holder() {}

    void init(type * log) {
        m_log = log;
        m_base = m_log->common_base();
    }
private:
    // note: I have a pointer to the log, as opposed to having it by value,
    // because the whole purpose of this class
    // is to be able to use a log without knowing its full type
    type *m_log;
    logger_base_type * m_base;
};



/**
    @brief Allows using a log without knowing its full type yet.
    Even if the log is not fully @b defined, you can still use it.

    This will allow you to log messages even if you don't know the full type of the log
    (which can aid compilation time).

    This keeps the logger by value, so that the after_being_destroyed stuff works.
    More specifically, in case the logger is used after it's been destroyed,
    the logger_holder instances CAN ONLY BE GLOBAL.
*/
template<class type> struct logger_holder_by_value : logger_holder<type> {
    typedef logger_holder<type> base_type;

    HPX_LOGGING_FORWARD_CONSTRUCTOR_INIT(logger_holder_by_value, m_log_value, init)
private:
    void init() {
        base_type::init( &m_log_value);
    }
private:
    // VERY IMPORTANT: we keep this BY VALUE, because,
    // at destruction, we don't want the memory to be freed
    // (in order for the after_being_destroyed to work,
    //  for global instances of this type)
    type m_log_value;
};


/**
    @brief Allows using a log without knowing its full type yet.
    Even if the log is not fully @b defined, you can still use it.

    This will allow you to log messages even if you don't know the full type of
    the log (which can aid compilation time).
*/
template<class type> struct logger_holder_by_ptr : logger_holder<type> {
    typedef logger_holder<type> base_type;

    HPX_LOGGING_FORWARD_CONSTRUCTOR_WITH_NEW_AND_INIT \
        (logger_holder_by_ptr, m_log_ptr, type, init)
    ~logger_holder_by_ptr() {
        delete m_log_ptr;
    }
private:
    void init() {
        base_type::init( m_log_ptr);
    }
private:
    type *m_log_ptr;
};







/**
    @brief Ensures the log is created before main(), even if not used before main

    We need this, so that we won't run into multi-threaded issues while
    the log is created
    (in other words, if the log is created before main(),
    we can safely assume there's only one thread running,
    thus no multi-threaded issues)
*/
struct ensure_early_log_creation {
    template<class type> ensure_early_log_creation ( type & log) {
    typedef boost::int64_t long_type ;
        long_type ignore = reinterpret_cast<long_type>(&log);
        // we need to force the compiler to force creation of the log
        if ( time(nullptr) < 0)
            if ( time(nullptr) < (time_t)ignore) {
                printf("LOGGING LIB internal error - should NEVER happen. \
                    Please report this to the author of the lib");
                exit(0);
            }
    }
};


/**
    @brief Ensures the filter is created before main(), even if not used before main

    We need this, so that we won't run into multi-threaded issues while
    the filter is created
    (in other words, if the filter is created before main(),
    we can safely assume there's only one thread running,
    thus no multi-threaded issues)
*/
typedef ensure_early_log_creation ensure_early_filter_creation;

/**
    Useful for logger_holder - to get the logger' base
    (so that we can use it even without knowing the full log's definition).

    If used on a logger, it just returns it .
*/
template<class logger> inline logger* get_logger_base(logger * l) { return l; }
template<class logger> inline const logger*
    get_logger_base(const logger * l) { return l; }

template<class type> inline typename logger_holder<type>::logger_base_type*
    get_logger_base(logger_holder<type> & l) {
    return l.base();
}
template<class type> inline const typename logger_holder<type>::logger_base_type*
    get_logger_base(const logger_holder<type> & l) {
    return l.base();
}


}}}

#endif


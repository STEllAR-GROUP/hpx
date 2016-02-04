// after_being_destroyed.hpp

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


#ifndef JT28092007_after_being_destroyed_HPP_DEFINED
#define JT28092007_after_being_destroyed_HPP_DEFINED

// see "Using the logger(s)/filter(s) after they've been destroyed" section in
// the documentation
#error this is obsolete

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>

#if !defined(HPX_LOG_AFTER_BEING_DESTROYED_WRITE_TO_FUNCTION) \
    && !defined(HPX_LOG_AFTER_BEING_DESTROYED_IGNORE) \
    && !defined(HPX_LOG_AFTER_BEING_DESTROYED_LEAK_LOGGER)
    // default
    #define HPX_LOG_AFTER_BEING_DESTROYED_WRITE_TO_FUNCTION
#endif

/**
    @file hpx/util/logging/detail/after_being_destoyed.hpp

    This file deals with the following situation:
    - what happens when someone is using the log(s) after they've been destroyed?
*/

namespace hpx { namespace util { namespace logging {

namespace destination { template<class T > struct msg_type; }

    /**
        deals with what to do with the logger, if used after it's been destroyed

    @remarks
        we need to make this a template, in order to postpone figuring
        out the gather_msg msg_type
        (so that we can wait until the user has specified the msg_type

    */
    template<class T = override> struct after_being_destroyed_defer_to_function {
        typedef typename destination::msg_type<T>::type type;
        typedef void (*after_destroyed_func)(const type&) ;

    protected:
        // default implementation - do nothing
        static void nothing(const type&) {}

        bool is_still_alive() const { return !m_is_destroyed; }
        void call_after_destroyed(const type& msg) const {
            m_after_being_destroyed(msg);
        }

        after_being_destroyed_defer_to_function () : m_is_destroyed(false),
            m_after_being_destroyed(&nothing) {}
        ~after_being_destroyed_defer_to_function () {
            m_is_destroyed = true;
        }
    private:
        bool m_is_destroyed;
    protected:
        after_destroyed_func m_after_being_destroyed;
    };

    template<class T = override> struct after_being_destroyed_none {
        typedef typename destination::msg_type<T>::type type;
        typedef void (*after_destroyed_func)(const type&) ;

    protected:
        after_being_destroyed_none () : m_after_being_destroyed(0) {}

        static bool is_still_alive() { return true; }
        // never called
        template<class type> void call_after_destroyed(const type&) const {}

    protected:
        // never used, needed for set_after_destroyed
        after_destroyed_func m_after_being_destroyed;
    };

    template<class T = override> struct after_being_destroyed
#ifdef HPX_LOG_AFTER_BEING_DESTROYED_WRITE_TO_FUNCTION
        : public after_being_destroyed_defer_to_function<T>
#elif defined(HPX_LOG_AFTER_BEING_DESTROYED_IGNORE)
        // ignore
        : public after_being_destroyed_none<T>
#elif defined(HPX_LOG_AFTER_BEING_DESTROYED_LEAK_LOGGER)
        // leaking, ignore this
        : public after_being_destroyed_none<T>
#endif
    {};

}}}

#endif


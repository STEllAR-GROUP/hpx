// cache_before_init_macros.hpp

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


#ifndef JT28092007_cache_before_init_macros_HPP_DEFINED
#define JT28092007_cache_before_init_macros_HPP_DEFINED

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#ifndef JT28092007_macros_HPP_DEFINED
#error Donot include this directly. Include hpx/util/logging/macros.hpp instead
#endif

#include <hpx/util/logging/detail/fwd.hpp>

namespace hpx { namespace util { namespace logging {

#if   defined(HPX_LOG_BEFORE_INIT_CACHE_FILTER)
    #define HPX_LOG_BEFORE_INIT_USE_CACHE_FILTER
#elif defined(HPX_LOG_BEFORE_INIT_LOG_ALL)
    #define HPX_LOG_BEFORE_INIT_USE_LOG_ALL
#elif defined(HPX_LOG_BEFORE_INIT_IGNORE_BEFORE_INIT)
    #define HPX_LOG_BEFORE_INIT_USE_IGNORE_BEFORE_INIT
#else
// use default
    #define HPX_LOG_BEFORE_INIT_USE_LOG_ALL
#endif




#if defined( HPX_LOG_BEFORE_INIT_USE_CACHE_FILTER)
////////////////////////////////////////////////////////////////////////////////////////
// Messages that were logged before initializing the log - cache the message & the filter

#define HPX_LOG_USE_LOG(l, do_func, is_log_enabled) \
    if (false) ; else struct local_class {                                          \
        static bool is_enabled_callback() { return (is_log_enabled); }              \
        local_class (const void * p) {                                              \
            if ( p)                                                                 \
                set_callback_if_needed();                                           \
        }                                                                           \
        void set_callback_if_needed() {                                             \
            if ( ::hpx::util::logging::get_logger_base( l )->is_cache_turned_off() )\
                ; \
            else \
                ::hpx::util::logging::get_logger_base( l )->cache().set_callback \
                  ( &is_enabled_callback ); \
        }                                                                           \
    } param = ( !(is_log_enabled) ) ? (void*)0 : \
                ::hpx::util::logging::get_logger_base( l )-> do_func


#elif defined( HPX_LOG_BEFORE_INIT_USE_LOG_ALL)
///////////////////////////////////////////////////////////////////////////////////
// Messages that were logged before initializing the log
// - cache the message (and I'll write it even if the filter is turned off)

#define HPX_LOG_USE_LOG(l, do_func, is_log_enabled) if ( !(is_log_enabled) ) ; \
        else ::hpx::util::logging::get_logger_base(l)-> do_func

#elif defined( HPX_LOG_BEFORE_INIT_USE_IGNORE_BEFORE_INIT)
//////////////////////////////////////////////////////////////////////////////////
// Messages that were logged before initializing the log - ignore them completely

#define HPX_LOG_USE_LOG(l, do_func, is_log_enabled) if ( !(is_log_enabled) ) ; \
        else ::hpx::util::logging::get_logger_base(l)-> do_func

#else
#error Internal error.
#endif

}}}

#endif


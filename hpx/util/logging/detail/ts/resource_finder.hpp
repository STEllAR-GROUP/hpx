// resource_finder.hpp

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


#ifndef JT28092007_resource_finder_HPP_DEFINED
#define JT28092007_resource_finder_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#ifndef JT28092007_ts_resource_HPP_DEFINED
#error never include directly, this is included by default
#endif

#include <hpx/util/logging/detail/fwd.hpp>

namespace hpx { namespace util { namespace logging {

    /**
        @brief Possible ways to lock resource for read/write
    */
    namespace lock_resource_finder {

    /**
        @brief Locks a resource thread-safe - each time,
        at read/write (safe but rather inefficient)
    */
    template<class mutex = hpx::util::logging::threading::mutex> struct ts {
            template<class lock_type> struct finder {
                typedef typename hpx::util::logging::locker
                    ::ts_resource<lock_type, mutex > type;
            };
    };

    /**
        @brief Does not lock the resouce at read/write access
    */
    struct single_thread {
            template<class lock_type> struct finder {
                typedef typename hpx::util::logging::locker
                    ::ts_resource_single_thread<lock_type> type;
            };
    };

#if !defined( HPX_HAVE_LOG_NO_TSS)
    /**
        @brief Caches the resource on each thread, and refreshes it at
        @c refresh_secs period
    */
    template<int refresh_secs = 5, class mutex = hpx::util::logging::threading::mutex >
    struct tss_with_cache {
            template<class lock_type> struct finder {
                typedef typename locker::tss_resource_with_cache<lock_type,
                    refresh_secs, mutex > type;
            };
    };

    /**
        @brief Allows you to initialize this resource once even if
        multiple threads are running. Then, all threads will use the initialized value
    */
    template<class mutex = hpx::util::logging::threading::mutex> struct tss_once_init {
            template<class lock_type> struct finder {
                typedef typename hpx::util::logging::locker::tss_resource_once_init
                    <lock_type, mutex> type;
            };
    };

#else

    // Not using TSS at all

    template<int = 5, class = hpx::util::logging::threading::mutex >
    struct tss_with_cache {
            template<class lock_type> struct finder {
                typedef typename locker::ts_resource_single_thread<lock_type> type;
            };
    };

    template<class = hpx::util::logging::threading::mutex> struct tss_once_init {
            template<class lock_type> struct finder {
                typedef typename locker::ts_resource_single_thread<lock_type> type;
            };
    };

#endif

}}}}

#endif


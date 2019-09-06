////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Thomas Heller
//  Copyright (c) 2008 Peter Dimov
//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#ifndef HPX_UTIL_DETAIL_YIELD_K_HPP
#define HPX_UTIL_DETAIL_YIELD_K_HPP

#include <hpx/config.hpp>

#ifdef HPX_HAVE_SPINLOCK_DEADLOCK_DETECTION
#include <hpx/errors/throw_exception.hpp>
#endif

#include <hpx/basic_execution/this_thread.hpp>

#include <hpx/runtime/threads/thread_enums.hpp>

namespace hpx { namespace util { namespace detail
{
#ifdef HPX_HAVE_SPINLOCK_DEADLOCK_DETECTION
    HPX_API_EXPORT extern bool spinlock_break_on_deadlock;
    HPX_API_EXPORT extern std::size_t spinlock_deadlock_detection_limit;
#endif

    inline void yield_k(std::size_t k, const char *thread_name,
        hpx::threads::thread_state_enum p = hpx::threads::pending_boost)
    {
#ifdef HPX_HAVE_SPINLOCK_DEADLOCK_DETECTION
        if (k > 32 && spinlock_break_on_deadlock &&
            k > spinlock_deadlock_detection_limit)
        {
            HPX_THROW_EXCEPTION(
                deadlock, thread_name, "possible deadlock detected");
        }
#endif
        hpx::basic_execution::this_thread::yield_k(k, thread_name);
    }
}}}

#endif

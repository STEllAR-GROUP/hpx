////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Thomas Heller
//  Copyright (c) 2008 Peter Dimov
//  Copyright (c) 2018 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#ifndef HPX_UTIL_DETAIL_YIELD_K_HPP
#define HPX_UTIL_DETAIL_YIELD_K_HPP

#include <hpx/config.hpp>

#include <hpx/runtime/threads/thread_helpers.hpp>
#ifdef HPX_HAVE_SPINLOCK_DEADLOCK_DETECTION
#include <hpx/throw_exception.hpp>
#endif

#include <chrono>
#include <cstddef>

#if defined (HPX_WINDOWS)
#include <windows.h>
#else
#  ifndef _AIX
#    include <sched.h>
#  else
    // AIX's sched.h defines ::var which sometimes conflicts with Lambda's var
    extern "C" int sched_yield(void);
#  endif
#  include <time.h>
#endif

namespace hpx { namespace util { namespace detail
{
#ifdef HPX_HAVE_SPINLOCK_DEADLOCK_DETECTION
    HPX_API_EXPORT extern bool spinlock_break_on_deadlock;
    HPX_API_EXPORT extern std::size_t spinlock_deadlock_detection_limit;
#endif

    inline void yield_k(std::size_t k, const char *thread_name,
        hpx::threads::thread_state_enum p = hpx::threads::pending_boost)
    {
        if (k < 4) //-V112
        {
        }
#if defined(HPX_SMT_PAUSE)
        else if (k < 16)
        {
            HPX_SMT_PAUSE;
        }
#endif
        else if (k < 32 || k & 1) //-V112
        {
            if (!hpx::threads::get_self_ptr())
            {
#if defined(HPX_WINDOWS)
                Sleep(0);
#else
                sched_yield();
#endif
            }
            else
            {
                hpx::this_thread::suspend(p, thread_name);
            }
        }
        else
        {
#ifdef HPX_HAVE_SPINLOCK_DEADLOCK_DETECTION
            if (spinlock_break_on_deadlock &&
                k > spinlock_deadlock_detection_limit)
            {
                HPX_THROW_EXCEPTION(deadlock,
                    thread_name, "possible deadlock detected");
            }
#endif
            if (!hpx::threads::get_self_ptr())
            {
#if defined(HPX_WINDOWS)
                Sleep(1);
#else
                // g++ -Wextra warns on {} or {0}
                struct timespec rqtp = { 0, 0 };

                // POSIX says that timespec has tv_sec and tv_nsec
                // But it doesn't guarantee order or placement

                rqtp.tv_sec = 0;
                rqtp.tv_nsec = 1000;

                nanosleep( &rqtp, nullptr );
#endif
            }
            else
            {
                hpx::this_thread::suspend(hpx::threads::pending, thread_name);
            }
        }
    }

}}}

#endif

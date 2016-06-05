////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Thomas Heller
//  Copyright (c) 2008 Peter Dimov
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#ifndef HPX_UTIL_DETAIL_YIELD_K_HPP
#define HPX_UTIL_DETAIL_YIELD_K_HPP

#include <hpx/config.hpp>

namespace hpx { namespace util { namespace detail {
    inline void yield_k(std::size_t k, const char *thread_name)
    {
        if (k < 4) //-V112
        {}
#if defined(BOOST_SMT_PAUSE)
        else if(k < 16)
        {
            BOOST_SMT_PAUSE
        }
#endif
        else if(k < 32 || k & 1) //-V112
        {
            if(!hpx::threads::get_self_ptr())
            {
#if defined(HPX_WINDOWS)
                Sleep(0);
#elif defined(BOOST_HAS_PTHREADS)
                sched_yield();
#else
#endif
            }
            else
            {
                hpx::this_thread::suspend(hpx::threads::pending, thread_name);
            }
        }
        else
        {
            if(!hpx::threads::get_self_ptr())
            {
#if defined(HPX_WINDOWS)
                Sleep(1);
#elif defined(BOOST_HAS_PTHREADS)
                // g++ -Wextra warns on {} or {0}
                struct timespec rqtp = { 0, 0 };

                // POSIX says that timespec has tv_sec and tv_nsec
                // But it doesn't guarantee order or placement

                rqtp.tv_sec = 0;
                rqtp.tv_nsec = 1000;

                nanosleep( &rqtp, 0 );
#else
#endif
            }
            else
            {
                hpx::this_thread::suspend(
                    boost::chrono::microseconds(1), thread_name);
            }
        }
    }

}}}

#endif

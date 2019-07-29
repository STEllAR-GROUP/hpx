//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_SET_THREAD_NAME_DEC_30_2012_1016AM)
#define HPX_UTIL_SET_THREAD_NAME_DEC_30_2012_1016AM

#include <hpx/config.hpp>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>

namespace hpx { namespace util {
    void set_thread_name(char const* threadName, DWORD dwThreadID = DWORD(-1));
}}    // namespace hpx::util

#else

namespace hpx { namespace util {
    inline void set_thread_name(char const* threadName) {}
}}    // namespace hpx::util

#endif

#endif

////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_UTIL_HARDWARE_TIMESTAMP_LINUX_GENERIC_HPP)
#define HPX_UTIL_HARDWARE_TIMESTAMP_LINUX_GENERIC_HPP

#if defined(__unix__)

#include <cstdint>

#include <time.h>

namespace hpx { namespace util { namespace hardware
{

inline std::uint64_t timestamp()
{
    struct timespec res;
    clock_gettime(CLOCK_MONOTONIC, &res);
    return 1000*res.tv_sec + res.tv_nsec/1000000;

}

}}}

#endif

#endif // HPX_UTIL_HARDWARE_TIMESTAMP_LINUX_GENERIC_HPP


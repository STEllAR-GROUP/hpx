////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>

#include <time.h>

#include <hpx/config.hpp>

namespace hpx { namespace util { namespace hardware {

    HPX_HOST_DEVICE inline std::uint64_t timestamp()
    {
#if defined(HPX_HAVE_CUDA) && defined(__CUDA_ARCH__)
        std::uint64_t cur;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(cur));
        return cur;
#else
        struct timespec res;
        clock_gettime(CLOCK_MONOTONIC, &res);
        return 1000 * res.tv_sec + res.tv_nsec / 1000000;
#endif
    }

}}}    // namespace hpx::util::hardware

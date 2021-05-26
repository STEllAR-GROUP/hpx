////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>
#include <hpx/config.hpp>

namespace hpx { namespace util { namespace hardware {

    // clang-format off
    HPX_HOST_DEVICE inline std::uint64_t timestamp()
    {
#if defined(HPX_HAVE_CUDA) && defined(__CUDA_ARCH__)
        std::uint64_t cur;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(cur));
        return cur;
#else
        std::uint32_t lo = 0, hi = 0;
        #if defined(HPX_HAVE_RDTSCP)
            __asm__ __volatile__(
                "rdtscp ;\n"
                : "=a"(lo), "=d"(hi)
                :
                : "rcx");
        #elif defined(HPX_HAVE_RDTSC)
            __asm__ __volatile__(
                "cpuid ;\n"
                "rdtsc ;\n"
                : "=a"(lo), "=d"(hi)
                :
                : "rbx", "rcx");
        #endif
        return ((static_cast<std::uint64_t>(hi)) << 32) | lo;
#endif
    }
    // clang-format on

}}}    // namespace hpx::util::hardware

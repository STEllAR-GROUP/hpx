////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>

#include <cstdint>

#if defined(HPX_HAVE_CUDA) && defined(HPX_COMPUTE_CODE)
#include <hpx/hardware/timestamp/cuda.hpp>
#endif

namespace hpx::util::hardware {

    // clang-format off
    [[nodiscard]] HPX_HOST_DEVICE inline std::uint64_t timestamp()
    {
#if defined(HPX_HAVE_CUDA) && defined(HPX_COMPUTE_DEVICE_CODE)
        return timestamp_cuda();
#else
        std::uint64_t r = 0;

#if defined(HPX_HAVE_RDTSCP)
        __asm__ __volatile__(
            "rdtscp ;\n"
            : "=A"(r)
            :
            : "%ecx");
#elif defined(HPX_HAVE_RDTSC)
        __asm__ __volatile__(
            "movl %%ebx, %%edi ;\n"
            "xorl %%eax, %%eax ;\n"
            "cpuid ;\n"
            "rdtsc ;\n"
            "movl %%edi, %%ebx ;\n"
            : "=A"(r)
            :
            : "%edi", "%ecx");
#endif

        return r;
#endif
    }
    // clang-format on
}    // namespace hpx::util::hardware

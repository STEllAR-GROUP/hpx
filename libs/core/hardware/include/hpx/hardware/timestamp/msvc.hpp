////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#if defined(HPX_WINDOWS)

#include <cstdint>

#include <intrin.h>
#include <windows.h>

#if defined(HPX_HAVE_CUDA)
#include <hpx/hardware/timestamp/cuda.hpp>
#endif

namespace hpx { namespace util { namespace hardware {
    inline std::uint64_t timestamp()
    {
#if defined(HPX_HAVE_CUDA) && defined(__CUDA_ARCH__)
        return timestamp_cuda();
#else
        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        return static_cast<std::uint64_t>(now.QuadPart);
#endif
    }
}}}    // namespace hpx::util::hardware

#endif

////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>

#include <time.h>

#include <cstdint>

#if defined(HPX_HAVE_CUDA) && defined(HPX_COMPUTE_CODE)
#include <hpx/hardware/timestamp/cuda.hpp>
#endif

namespace hpx::util::hardware {

    [[nodiscard]] HPX_HOST_DEVICE inline std::uint64_t timestamp()
    {
#if defined(HPX_HAVE_CUDA) && defined(HPX_COMPUTE_DEVICE_CODE)
        return timestamp_cuda();
#else
        struct timespec res;
        clock_gettime(CLOCK_MONOTONIC, &res);
        return 1000 * res.tv_sec + res.tv_nsec / 1000000;
#endif
    }
}    // namespace hpx::util::hardware

////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#if defined(__bgq__)

// Hardware cycle-accurate timer on BGQ.
// see https://wiki.alcf.anl.gov/parts/index.php/Blue_Gene/Q#High-Resolution_Timers

#include <hwi/include/bqc/A2_inlines.h>

#include <hpx/config.hpp>

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
        return GetTimeBase();
#endif
    }
}    // namespace hpx::util::hardware

#endif

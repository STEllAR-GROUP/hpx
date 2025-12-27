//  Copyright (c) 2011-2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//  Copyright (c)      2013 Patricia Grubel
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/modules/timing.hpp>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

HPX_FORCEINLINE void worker_timed(std::uint64_t delay_ns) noexcept
{
    if (delay_ns == 0)
        return;

    using clock = hpx::chrono::high_resolution_clock;

    auto const end = clock::now() + delay_ns;

    while (clock::now() < end)
    {
#if defined(__x86_64__) || defined(_M_X64)
        _mm_pause();
#endif
    }
}

//  Copyright (c) 2011-2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//  Copyright (c)      2013 Patricia Grubel
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TESTS_PERFORMANCE_WORKER_HPP
#define HPX_TESTS_PERFORMANCE_WORKER_HPP

#include <hpx/util/high_resolution_clock.hpp>

#include <cstdint>

inline void worker_timed(std::uint64_t delay_ns)
{
    if (delay_ns == 0)
        return;

    std::uint64_t start = hpx::util::high_resolution_clock::now();

    while (true)
    {
        // Check if we've reached the specified delay.
        if ((hpx::util::high_resolution_clock::now() - start) >= delay_ns)
            break;
    }
}

#endif

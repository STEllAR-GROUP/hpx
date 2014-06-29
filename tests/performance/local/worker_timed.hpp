//  Copyright (c) 2011-2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//  Copyright (c)      2013 Patricia Grubel
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TESTS_PERFORMANCE_WORKER_HPP
#define HPX_TESTS_PERFORMANCE_WORKER_HPP

//#include <hpx/config.hpp>
#include <hpx/util/hardware/timestamp.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <boost/cstdint.hpp>

inline void worker_timed(
    boost::uint64_t delay_ns
    )
{
    if (delay_ns == 0)
        return;

    using namespace hpx::util;

    boost::uint64_t start = high_resolution_clock::now();

    while (true)
    {
        // Check if we've reached the specified delay.
        if ((high_resolution_clock::now() - start) >= delay_ns)
            break;
    }
}

#endif

//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OPENMP_GRID
#include <hpx/hpx_fwd.hpp>
#endif
#include "grid.hpp"

#ifndef OPENMP_GRID
#include <hpx/runtime/actions/plain_action.hpp>

std::size_t touch_mem(std::size_t desired, std::size_t ps, std::size_t start, std::size_t end)
{
    std::size_t current = hpx::get_worker_thread_num();

    if (current == desired)
    {
        // Yes! The PX-thread is run by the designated OS-thread.
        char * p = reinterpret_cast<char *>(ps);
        for(std::size_t i = start; i < end; ++i)
        {
            p[i] = 0;
        }
        return desired;
    }

    // this PX-thread is run by the wrong OS-thread, make the foreman retry
    return std::size_t(-1);
}

#endif

namespace bright_future
{
}

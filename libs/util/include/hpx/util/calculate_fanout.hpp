//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_CALCULATE_FANOUT_APR_23_2014_0124PM)
#define HPX_UTIL_CALCULATE_FANOUT_APR_23_2014_0124PM

#include <cstddef>

namespace hpx { namespace util
{
    inline std::size_t
    calculate_fanout(std::size_t size, std::size_t local_fanout)
    {
        if (size == 0 || local_fanout == 0)
            return 1;
        if (size <= local_fanout)
            return size;

        std::size_t fanout = 1;
        size -= local_fanout;
        while (fanout < size)
        {
            fanout *= local_fanout;
        }
        return fanout;
    }
}}

#endif

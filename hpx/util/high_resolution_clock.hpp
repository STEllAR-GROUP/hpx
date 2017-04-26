//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_HIGH_RESOLUTION_CLOCK_FEB_24_2012_1125AM)
#define HPX_HIGH_RESOLUTION_CLOCK_FEB_24_2012_1125AM

#include <hpx/config.hpp>

#if defined(__bgq__)
#include <hwi/include/bqc/A2_inlines.h>
#endif

#include <chrono>
#include <cstdint>

namespace hpx { namespace util
{
    struct high_resolution_clock
    {
        // This function returns a tick count with a resolution (not
        // precision!) of 1 ns.
        static std::uint64_t now()
        {
#if defined(__bgq__)
            return GetTimeBase();
#else
            std::chrono::nanoseconds ns =
                std::chrono::steady_clock::now().time_since_epoch();
            return static_cast<std::uint64_t>(ns.count());
#endif
        }

        // This function returns the smallest representable time unit as
        // returned by this clock.
        static std::uint64_t (min)()
        {
            typedef std::chrono::duration_values<std::chrono::nanoseconds>
                duration_values;
            return (duration_values::min)().count();
        }

        // This function returns the largest representable time unit as
        // returned by this clock.
        static std::uint64_t (max)()
        {
            typedef std::chrono::duration_values<std::chrono::nanoseconds>
                duration_values;
            return (duration_values::max)().count();
        }
    };
}}

#endif

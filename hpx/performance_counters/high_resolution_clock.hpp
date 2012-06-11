//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_HIGH_RESOLUTION_CLOCK_FEB_24_2012_1125AM)
#define HPX_HIGH_RESOLUTION_CLOCK_FEB_24_2012_1125AM

#include <boost/chrono/chrono.hpp>
#include <boost/chrono/process_cpu_clocks.hpp>

namespace hpx { namespace performance_counters
{
    struct high_resolution_clock
    {
        // This function returns a tick count with a resolution (not
        // precision!) of 1 ns.
        static boost::uint64_t now()
        {
            boost::chrono::nanoseconds ns =
                boost::chrono::high_resolution_clock::now().time_since_epoch();
            BOOST_ASSERT(ns.count() >= 0);
            return static_cast<boost::uint64_t>(ns.count());
        }
    };
}}

#endif

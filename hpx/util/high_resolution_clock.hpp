//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_HIGH_RESOLUTION_CLOCK_FEB_24_2012_1125AM)
#define HPX_HIGH_RESOLUTION_CLOCK_FEB_24_2012_1125AM

#include <boost/chrono/chrono.hpp>
#include <boost/chrono/process_cpu_clocks.hpp>
#include <boost/assert.hpp>

namespace hpx { namespace util
{
    struct high_resolution_clock
    {
        // This function returns a tick count with a resolution (not
        // precision!) of 1 ns.
        static boost::uint64_t now()
        {
            boost::chrono::nanoseconds ns =
                boost::chrono::steady_clock::now().time_since_epoch();
            BOOST_ASSERT(ns.count() >= 0);
            return static_cast<boost::uint64_t>(ns.count());
        }

        // This function returns the smallest representable time unit as 
        // returned by this clock.
        static boost::uint64_t (min)()
        {
            typedef boost::chrono::duration_values<boost::chrono::nanoseconds>
                duration_values;
            return (duration_values::min)().count();
        }

        // This function returns the largest representable time unit as 
        // returned by this clock.
        static boost::uint64_t (max)()
        {
            typedef boost::chrono::duration_values<boost::chrono::nanoseconds>
                duration_values;
            return (duration_values::max)().count();
        }
    };
}}

#endif

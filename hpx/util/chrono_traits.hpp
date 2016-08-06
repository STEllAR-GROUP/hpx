//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_CHRONO_TRAITS_HPP
#define HPX_UTIL_CHRONO_TRAITS_HPP

#include <hpx/config.hpp>

#include <boost/date_time/posix_time/posix_time_duration.hpp>

#include <chrono>

namespace hpx { namespace util
{
    template <typename Clock>
    struct chrono_traits
    {
        typedef typename Clock::duration duration_type;
        typedef typename Clock::time_point time_type;

        static time_type now() HPX_NOEXCEPT
        {
            return Clock::now();
        }

        static time_type add(time_type t, duration_type d)
        {
            return t + d;
        }

        static duration_type subtract(time_type t1, time_type t2)
        {
            return t1 - t2;
        }

        static bool less_than(time_type t1, time_type t2)
        {
            return t1 < t2;
        }

        static boost::posix_time::time_duration to_posix_duration(duration_type d)
        {
#ifdef BOOST_DATE_TIME_HAS_NANOSECONDS
            return boost::posix_time::nanoseconds(
                std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
#else
            return boost::posix_time::microseconds(
                std::chrono::duration_cast<std::chrono::microseconds>(d).count());
#endif
        }
    };
}}

#endif /*HPX_UTIL_CHRONO_TRAITS_HPP*/

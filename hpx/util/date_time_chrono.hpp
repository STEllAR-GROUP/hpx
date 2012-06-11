//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_DATE_TIME_CHRONO_APR_10_2012_0449PM)
#define HPX_UTIL_DATE_TIME_CHRONO_APR_10_2012_0449PM

#include <hpx/hpx_fwd.hpp>
#include <boost/chrono/chrono.hpp>
#include <boost/date_time/posix_time/posix_time_duration.hpp>
#include <boost/date_time/posix_time/ptime.hpp>

namespace hpx { namespace util
{
    template <typename Clock, typename Duration>
    boost::posix_time::ptime
    to_ptime(boost::chrono::time_point<Clock, Duration> const& from)
    {
        typedef boost::chrono::time_point<Clock, Duration> time_point_type;
        typedef boost::chrono::nanoseconds duration_type;
        typedef duration_type::rep rep_type;
        rep_type d = boost::chrono::duration_cast<duration_type>(
            from.time_since_epoch()).count();
        rep_type sec = d / 1000000000;
        rep_type nsec = d % 1000000000;
        return boost::posix_time::from_time_t(0) +

            boost::posix_time::seconds(static_cast<long>(sec)) +
#ifdef BOOST_DATE_TIME_HAS_NANOSECONDS
            boost::posix_time::nanoseconds(nsec);
#else
            boost::posix_time::microseconds((nsec+500)/1000);
#endif
    }

    template <typename Clock, typename Duration>
    boost::posix_time::time_duration
    to_time_duration(boost::chrono::duration<Clock, Duration> const& from)
    {
        typedef boost::chrono::nanoseconds duration_type;
        typedef duration_type::rep rep_type;
        rep_type d = boost::chrono::duration_cast<duration_type>(
            from.count()).count();
        rep_type sec = d / 1000000000;
        rep_type nsec = d % 1000000000;
        return boost::posix_time::seconds(static_cast<long>(sec)) +
#ifdef BOOST_DATE_TIME_HAS_NANOSECONDS
            boost::posix_time::nanoseconds(nsec);
#else
            boost::posix_time::microseconds((nsec+500)/1000);
#endif
    }

//     template <typename Clock, typename Duration>
//     boost::chrono::time_point<Clock, Duration>
//     to_time_point(boost::posix_time::ptime const& from)
//     {
//         boost::posix_time::time_duration const time_since_epoch =
//             from - boost::posix_time::from_time_t(0);
//         boost::chrono::time_point<Clock, Duration> t =
//           boost::chrono::system_clock::from_time_t(time_since_epoch.total_seconds());
//         long nsec = time_since_epoch.fractional_seconds() *
//             (1000000000 / time_since_epoch.ticks_per_second());
//         return t + boost::chrono::nanoseconds(nsec);
//     }
}}

#endif

//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_DATE_TIME_CHRONO_APR_10_2012_0449PM)
#define HPX_UTIL_DATE_TIME_CHRONO_APR_10_2012_0449PM

#include <hpx/config.hpp>
#include <boost/chrono/chrono.hpp>
#include <boost/date_time/posix_time/conversion.hpp>
#include <boost/date_time/posix_time/posix_time_duration.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/thread/thread_time.hpp>

#include <chrono>

namespace hpx { namespace util
{
    template <typename Clock, typename Duration>
    boost::posix_time::ptime
    to_ptime(std::chrono::time_point<Clock, Duration> const& from)
    {
        typedef std::chrono::nanoseconds duration_type;
        typedef duration_type::rep rep_type;
        rep_type d = std::chrono::duration_cast<duration_type>(
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
    to_time_duration(std::chrono::duration<Clock, Duration> const& from)
    {
        typedef std::chrono::nanoseconds duration_type;
        typedef duration_type::rep rep_type;
        rep_type d = std::chrono::duration_cast<duration_type>(from).count();
        rep_type sec = d / 1000000000;
        rep_type nsec = d % 1000000000;
        return boost::posix_time::seconds(static_cast<long>(sec)) +
#ifdef BOOST_DATE_TIME_HAS_NANOSECONDS
            boost::posix_time::nanoseconds(nsec);
#else
            boost::posix_time::microseconds((nsec+500)/1000);
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    struct posix_clock
    {
        typedef boost::int_least64_t rep;
        typedef std::nano period;
        typedef std::chrono::duration<rep, period> duration;
        typedef std::chrono::time_point<posix_clock, duration> time_point;
        HPX_STATIC_CONSTEXPR bool is_steady = false;

        static HPX_FORCEINLINE time_point now() HPX_NOEXCEPT
        {
            return from_ptime(boost::get_system_time());
        }

        static boost::posix_time::ptime to_ptime(time_point const& t) HPX_NOEXCEPT
        {
            return boost::posix_time::from_time_t(0) +
#if defined(BOOST_DATE_TIME_HAS_NANOSECONDS)
                boost::posix_time::nanoseconds(t.time_since_epoch().count());
#else
                boost::posix_time::microseconds(t.time_since_epoch().count() / 1000);
#endif
        }

        static time_point from_ptime(boost::posix_time::ptime const& t) HPX_NOEXCEPT
        {
            return time_point(duration(
              (t - boost::posix_time::from_time_t(0)).total_nanoseconds()));
        }
    };

    class steady_time_point
    {
        typedef std::chrono::steady_clock::time_point value_type;

    public:
        steady_time_point(value_type const& abs_time)
          : _abs_time(abs_time)
        {}

        template <typename Clock, typename Duration>
        steady_time_point(std::chrono::time_point<Clock, Duration> const& abs_time)
          : _abs_time(std::chrono::steady_clock::now()
              + (abs_time - Clock::now()))
        {}

        template <typename Clock, typename Duration>
        steady_time_point(boost::chrono::time_point<Clock, Duration> const& abs_time)
        {
            std::chrono::nanoseconds duration(
                boost::chrono::duration_cast<boost::chrono::nanoseconds>(
                    abs_time - Clock::now()).count());

            _abs_time = std::chrono::steady_clock::now() + duration;
        }

        steady_time_point(boost::posix_time::ptime const& abs_time)
          : _abs_time(std::chrono::steady_clock::now()
              + (posix_clock::from_ptime(abs_time) - posix_clock::now()))
        {}

        value_type const& value() const HPX_NOEXCEPT
        {
            return _abs_time;
        }

    private:
        value_type _abs_time;
    };

    class steady_duration
    {
        typedef std::chrono::steady_clock::duration value_type;

    public:
        steady_duration(value_type const& rel_time)
          : _rel_time(rel_time)
        {}

        template <typename Rep, typename Period>
        steady_duration(std::chrono::duration<Rep, Period> const& rel_time)
          : _rel_time(std::chrono::duration_cast<value_type>(rel_time))
        {
            if (_rel_time < rel_time)
                ++_rel_time;
        }

        steady_duration(boost::posix_time::time_duration const& rel_time)
          : _rel_time(rel_time.total_nanoseconds())
        {}

        template <typename Rep, typename Period>
        steady_duration(boost::chrono::duration<Rep, Period> const& rel_time)
        {
            std::chrono::nanoseconds duration(
                boost::chrono::duration_cast<boost::chrono::nanoseconds>(
                    rel_time).count());

            _rel_time = std::chrono::duration_cast<value_type>(duration);
            if (_rel_time < duration)
                ++_rel_time;
        }

        value_type const& value() const HPX_NOEXCEPT
        {
            return _rel_time;
        }

        std::chrono::steady_clock::time_point from_now() const HPX_NOEXCEPT
        {
            return std::chrono::steady_clock::now() + _rel_time;
        }

    private:
        value_type _rel_time;
    };

    ///////////////////////////////////////////////////////////////////////////
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

#endif

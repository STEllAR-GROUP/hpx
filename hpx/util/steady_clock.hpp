//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2014-2016 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedinclude:boost/chrono/chrono.hpp
// hpxinspect:nodeprecatedname:boost::chrono

#ifndef HPX_UTIL_STEADY_CLOCK_HPP
#define HPX_UTIL_STEADY_CLOCK_HPP

#include <hpx/config.hpp>

#include <chrono>

namespace hpx { namespace util
{
    using std::chrono::steady_clock;

    class steady_time_point
    {
        typedef steady_clock::time_point value_type;

    public:
        steady_time_point(value_type const& abs_time)
          : _abs_time(abs_time)
        {}

        template <typename Clock, typename Duration>
        steady_time_point(std::chrono::time_point<Clock, Duration> const& abs_time)
          : _abs_time(std::chrono::time_point_cast<value_type::duration>(
                steady_clock::now() + (abs_time - Clock::now())))
        {}

        value_type const& value() const noexcept
        {
            return _abs_time;
        }

    private:
        value_type _abs_time;
    };

    class steady_duration
    {
        typedef steady_clock::duration value_type;

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

        value_type const& value() const noexcept
        {
            return _rel_time;
        }

        steady_clock::time_point from_now() const noexcept
        {
            return steady_clock::now() + _rel_time;
        }

    private:
        value_type _rel_time;
    };
}}

#endif /*HPX_UTIL_STEADY_CLOCK_HPP*/

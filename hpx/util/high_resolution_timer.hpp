//  Copyright (c) 2005-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_HIGH_RESOLUTION_TIMER_MAR_24_2008_1222PM)
#define HPX_UTIL_HIGH_RESOLUTION_TIMER_MAR_24_2008_1222PM

#include <hpx/config.hpp>
#include <hpx/util/high_resolution_clock.hpp>

#include <boost/chrono/chrono.hpp>
#include <boost/cstdint.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    //
    //  high_resolution_timer
    //      A timer object measures elapsed time.
    //
    ///////////////////////////////////////////////////////////////////////////
    class high_resolution_timer
    {
    public:
        high_resolution_timer()
          : start_time_(take_time_stamp())
        {
        }

        high_resolution_timer(double t)
          : start_time_(static_cast<boost::uint64_t>(t * 1e9))
        {}

        static double now()
        {
            return take_time_stamp() * 1e-9;
        }

        void restart()
        {
            start_time_ = take_time_stamp();
        }
        double elapsed() const                  // return elapsed time in seconds
        {
            return double(take_time_stamp() - start_time_) * 1e-9;
        }

        boost::int64_t elapsed_microseconds() const
        {
            return boost::int64_t((take_time_stamp() - start_time_) * 1e-3);
        }

        boost::int64_t elapsed_nanoseconds() const
        {
            return boost::int64_t(take_time_stamp() - start_time_);
        }

        double elapsed_max() const   // return estimated maximum value for elapsed()
        {
            return (util::high_resolution_clock::max)() * 1e-9;
        }

        double elapsed_min() const   // return minimum value for elapsed()
        {
            return (util::high_resolution_clock::min)() * 1e-9;
        }

    protected:
        static boost::uint64_t take_time_stamp()
        {
            return util::high_resolution_clock::now();
        }

    private:
        boost::uint64_t start_time_;
    };
}} // namespace hpx::util

#endif


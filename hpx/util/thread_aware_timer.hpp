//  Copyright (c) 2005-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_THREAD_AWARE_TIMER_AUG_17_2012_0745PM)
#define HPX_UTIL_THREAD_AWARE_TIMER_AUG_17_2012_0745PM

#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/promise.hpp>
#include <hpx/timing/high_resolution_clock.hpp>

#include <cstdint>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////////
    //
    //  thread_aware_timer
    //      A timer object measures elapsed time making sure the samples are taken
    //      from the same os-thread.
    //
    ///////////////////////////////////////////////////////////////////////////////
    class thread_aware_timer
    {
    public:
        thread_aware_timer()
          : start_time_(take_time_stamp())
        {
        }

        thread_aware_timer(double t)
          : start_time_(static_cast<std::uint64_t>(t * 1e9))
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

        std::int64_t elapsed_microseconds() const
        {
            return std::int64_t((take_time_stamp() - start_time_) * 1e-3);
        }

        std::int64_t elapsed_nanoseconds() const
        {
            return std::int64_t(take_time_stamp() - start_time_);
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
        static void sample_time(hpx::lcos::local::promise<std::uint64_t>& p)
        {
            p.set_value(util::high_resolution_clock::now());
        }

        HPX_EXPORT static std::uint64_t take_time_stamp();

    private:
        std::uint64_t start_time_;
    };
}} // namespace hpx::util

#endif


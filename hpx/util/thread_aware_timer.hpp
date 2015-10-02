//  Copyright (c) 2005-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_THREAD_AWARE_TIMER_AUG_17_2012_0745PM)
#define HPX_UTIL_THREAD_AWARE_TIMER_AUG_17_2012_0745PM

#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/promise.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/util/io_service_pool.hpp>

#include <boost/cstdint.hpp>

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
        static void sample_time(hpx::lcos::local::promise<boost::uint64_t>& p)
        {
            p.set_value(util::high_resolution_clock::now());
        }

        static boost::uint64_t take_time_stamp()
        {
            hpx::lcos::local::promise<boost::uint64_t> p;

            // Get a reference to the Timer specific HPX io_service object ...
            hpx::util::io_service_pool* pool = hpx::get_thread_pool("timer_pool");

            // ... and schedule the handler to run on the first of its OS-threads.
            pool->get_io_service(0).post(hpx::util::bind(&sample_time, boost::ref(p)));
            return p.get_future().get();
        }

    private:
        boost::uint64_t start_time_;
    };
}} // namespace hpx::util

#endif


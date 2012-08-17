//  Copyright (c) 2005-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_HIGH_RESOLUTION_TIMER_MAR_24_2008_1222PM)
#define HPX_UTIL_HIGH_RESOLUTION_TIMER_MAR_24_2008_1222PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/util/high_resolution_clock.hpp>

#include <boost/config.hpp>
#include <boost/throw_exception.hpp>
#include <boost/cstdint.hpp>
#include <boost/chrono/chrono.hpp>

#include <stdexcept>

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
          : start_time_(now())
        {
        }

        thread_aware_timer(double t)
          : start_time_(t * 1e9)
        {}

        thread_aware_timer(thread_aware_timer const& rhs)
          : start_time_(rhs.start_time_)
        {}

        static double now()
        {
            return take_time_stamp() * 1e-9;
        }

        void restart()
        {
            start_time_ = now();
        }
        double elapsed() const                  // return elapsed time in seconds
        {
            return double(now() - start_time) * 1e-9;
        }

        boost::int64_t elapsed_microseconds() const
        {
            return boost::int64_t((now() - start_time) * 1e-3);
        }

        boost::int64_t elapsed_nanoseconds() const
        {
            return boost::int64_t(now() - start_time);
        }

        double elapsed_max() const   // return estimated maximum value for elapsed()
        {
            return (boost::chrono::duration_values<boost::chrono::nanoseconds>::max)() * 1e-9;
        }

        double elapsed_min() const   // return minimum value for elapsed()
        {
            return (boost::chrono::duration_values<boost::chrono::nanoseconds>::min)() * 1e-9;
        }

    protected:
        static void sample_time(
            boost::shared_ptr<hpx::lcos::local::promise<boost::uint64_t> > p)
        {
            p->set_value(util::high_resolution_clock::now());
        }

        static boost::uint64_t take_time_stamp()
        {
            boost::shared_ptr<hpx::lcos::local::promise<boost::uint64_t> > p =
                boost::make_shared<hpx::lcos::local::promise<boost::uint64_t> >();

            // Get a reference to the Timer specific HPX io_service object ...
            hpx::util::io_service_pool* pool =
                hpx::get_runtime().get_thread_pool("timer_pool");

            // ... and schedule the handler to run on the first of its OS-threads.
            pool->get_io_service(0).post(hpx::util::bind(&sample_time, p));
            return p->get_future().get();
        }

    private:
        boost::uint64_t start_time_;
    };
}} // namespace hpx::util

#endif


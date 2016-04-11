//  Copyright (c) 2005-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/util/thread_aware_timer.hpp>
#include <hpx/util/io_service_pool.hpp>

#include <boost/cstdint.hpp>

namespace hpx { namespace util
{
    boost::uint64_t thread_aware_timer::take_time_stamp()
    {
        hpx::lcos::local::promise<boost::uint64_t> p;

        // Get a reference to the Timer specific HPX io_service object ...
        hpx::util::io_service_pool* pool = hpx::get_thread_pool("timer_pool");

        // ... and schedule the handler to run on the first of its OS-threads.
        pool->get_io_service(0).post(hpx::util::bind(&sample_time, boost::ref(p)));
        return p.get_future().get();
    }
}}


//  Copyright (c) 2005-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THREAD_AWARE_TIMER_COMPATIBILITY)
#include <hpx/functional/bind_front.hpp>
#include <hpx/io_service/io_service_pool.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/util/thread_aware_timer.hpp>

#include <cstdint>

#include <functional>

namespace hpx { namespace util {
    std::uint64_t thread_aware_timer::take_time_stamp()
    {
        hpx::lcos::local::promise<std::uint64_t> p;

        // Get a reference to the Timer specific HPX io_service object ...
        hpx::util::io_service_pool* pool = hpx::get_thread_pool("timer_pool");

        // ... and schedule the handler to run on the first of its OS-threads.
        pool->get_io_service(0).post(
            hpx::util::bind_front(&sample_time, std::ref(p)));
        return p.get_future().get();
    }
}}    // namespace hpx::util

#endif

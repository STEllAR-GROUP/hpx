//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2007-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c)      2011 Bryce Lelbach, Katelyn Kufahl
//  Copyright (c)      2017 Shoshana Jakobovits
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/performance_counters/counters_fwd.hpp>
#include <hpx/modules/threadmanager.hpp>

#include <cstddef>
#include <cstdint>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads {
    namespace detail {
        // counter creator functions
        naming::gid_type thread_counts_counter_creator(
            performance_counters::counter_info const& info, error_code& ec);
        naming::gid_type scheduler_utilization_counter_creator(
            threadmanager* tm, performance_counters::counter_info const& info,
            error_code& ec);

        typedef std::int64_t (threadmanager::*threadmanager_counter_func)(
            bool reset);
        typedef std::int64_t (thread_pool_base::*threadpool_counter_func)(
            std::size_t num_thread, bool reset);

        naming::gid_type locality_pool_thread_counter_creator(threadmanager* tm,
            threadmanager_counter_func total_func,
            threadpool_counter_func pool_func,
            performance_counters::counter_info const& info, error_code& ec);

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
        naming::gid_type queue_wait_time_counter_creator(threadmanager* tm,
            threadmanager_counter_func total_func,
            threadpool_counter_func pool_func,
            performance_counters::counter_info const& info, error_code& ec);
#endif

        naming::gid_type locality_pool_thread_no_total_counter_creator(
            threadmanager* tm, threadpool_counter_func pool_func,
            performance_counters::counter_info const& info, error_code& ec);
    }

    HPX_EXPORT void register_counter_types(threadmanager& tm);
}}

#include <hpx/config/warnings_suffix.hpp>


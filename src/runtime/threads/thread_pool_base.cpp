//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/compat/thread.hpp>
#include <hpx/compat/mutex.hpp>
#include <hpx/error_code.hpp>
#include <hpx/exception.hpp>
#include <hpx/exception_info.hpp>
#include <hpx/state.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/runtime/get_worker_thread_num.hpp>
#include <hpx/runtime/threads/detail/set_thread_state.hpp>
#include <hpx/runtime/threads/policies/callback_notifier.hpp>
#include <hpx/runtime/threads/thread_pool_base.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/hardware/timestamp.hpp>
#include <hpx/util/high_resolution_clock.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    thread_pool_base::thread_pool_base(
            threads::policies::callback_notifier& notifier,
            std::size_t index, std::string const& pool_name,
            policies::scheduler_mode m, std::size_t thread_offset)
      : id_(index, pool_name),
        mode_(m),
        thread_offset_(thread_offset),
        timestamp_scale_(1.0),
        notifier_(notifier)
    {}

    ///////////////////////////////////////////////////////////////////////////
    mask_type thread_pool_base::get_used_processing_units() const
    {
        auto const& rp = resource::get_partitioner();
        auto const sched = get_scheduler();

        mask_type used_processing_units = mask_type();
        threads::resize(used_processing_units, hardware_concurrency());

        for (std::size_t thread_num = 0; thread_num < get_os_thread_count();
            ++thread_num)
        {
            if (sched->get_state(thread_num).load() <= state_suspended)
            {
                used_processing_units |=
                    rp.get_pu_mask(thread_num + get_thread_offset());
            }
        }

        return used_processing_units;
    }

    hwloc_bitmap_ptr thread_pool_base::get_numa_domain_bitmap() const
    {
        auto const& rp = resource::get_partitioner();
        mask_type used_processing_units = get_used_processing_units();
        return rp.get_topology().cpuset_to_nodeset(used_processing_units);
    }

    std::size_t thread_pool_base::get_active_os_thread_count() const
    {
        std::size_t active_os_thread_count = 0;

        for (std::size_t thread_num = 0; thread_num < get_os_thread_count();
            ++thread_num)
        {
            if (get_scheduler()->get_state(thread_num).load() <= state_suspended)
            {
                ++active_os_thread_count;
            }
        }

        return active_os_thread_count;
    }

    ///////////////////////////////////////////////////////////////////////////
    // detail::manage_executor interface implementation
    char const* thread_pool_base::get_description() const
    {
        return id_.name().c_str();
    }

    ///////////////////////////////////////////////////////////////////////////
    void thread_pool_base::init_pool_time_scale()
    {
        // scale timestamps to nanoseconds
        std::uint64_t base_timestamp = util::hardware::timestamp();
        std::uint64_t base_time = util::high_resolution_clock::now();
        std::uint64_t curr_timestamp = util::hardware::timestamp();
        std::uint64_t curr_time = util::high_resolution_clock::now();

        while ((curr_time - base_time) <= 100000)
        {
            curr_timestamp = util::hardware::timestamp();
            curr_time = util::high_resolution_clock::now();
        }

        if (curr_timestamp - base_timestamp != 0)
        {
            timestamp_scale_ = double(curr_time - base_time) /
                double(curr_timestamp - base_timestamp);
        }
    }

    void thread_pool_base::init(std::size_t pool_threads,
        std::size_t threads_offset)
    {
        thread_offset_ = threads_offset;
    }
}}


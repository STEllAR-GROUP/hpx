//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/affinity/affinity_data.hpp>
#include <hpx/assert.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/hardware/timestamp.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/threading_base/callback_notifier.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/scheduler_state.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>
#include <hpx/timing/high_resolution_clock.hpp>
#include <hpx/topology/topology.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace hpx { namespace threads {
    ///////////////////////////////////////////////////////////////////////////
    thread_pool_base::thread_pool_base(thread_pool_init_parameters const& init)
      : id_(init.index_, init.name_)
      , thread_offset_(init.thread_offset_)
      , affinity_data_(init.affinity_data_)
      , timestamp_scale_(1.0)
      , notifier_(init.notifier_)
    {
    }

    ///////////////////////////////////////////////////////////////////////////
    mask_type thread_pool_base::get_used_processing_units() const
    {
        auto const& topo = create_topology();
        auto const sched = get_scheduler();

        mask_type used_processing_units = mask_type();
        threads::resize(used_processing_units, hardware_concurrency());

        for (std::size_t thread_num = 0; thread_num < get_os_thread_count();
             ++thread_num)
        {
            if (sched->get_state(thread_num).load() <= state_suspended)
            {
                used_processing_units |= affinity_data_.get_pu_mask(
                    topo, thread_num + get_thread_offset());
            }
        }

        return used_processing_units;
    }

    hwloc_bitmap_ptr thread_pool_base::get_numa_domain_bitmap() const
    {
        auto const& topo = create_topology();
        mask_type used_processing_units = get_used_processing_units();
        return topo.cpuset_to_nodeset(used_processing_units);
    }

    std::size_t thread_pool_base::get_active_os_thread_count() const
    {
        std::size_t active_os_thread_count = 0;

        for (std::size_t thread_num = 0; thread_num < get_os_thread_count();
             ++thread_num)
        {
            if (get_scheduler()->get_state(thread_num).load() <=
                state_suspended)
            {
                ++active_os_thread_count;
            }
        }

        return active_os_thread_count;
    }

#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
    ///////////////////////////////////////////////////////////////////////////
    // detail::manage_executor interface implementation
    char const* thread_pool_base::get_description() const
    {
        return id_.name().c_str();
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    void thread_pool_base::init_pool_time_scale()
    {
        // scale timestamps to nanoseconds
        std::uint64_t base_timestamp = util::hardware::timestamp();
        std::uint64_t base_time = hpx::chrono::high_resolution_clock::now();
        std::uint64_t curr_timestamp = util::hardware::timestamp();
        std::uint64_t curr_time = hpx::chrono::high_resolution_clock::now();

        while ((curr_time - base_time) <= 100000)
        {
            curr_timestamp = util::hardware::timestamp();
            curr_time = hpx::chrono::high_resolution_clock::now();
        }

        if (curr_timestamp - base_timestamp != 0)
        {
            timestamp_scale_ = double(curr_time - base_time) /
                double(curr_timestamp - base_timestamp);
        }
    }

    void thread_pool_base::init(
        std::size_t /* pool_threads */, std::size_t threads_offset)
    {
        thread_offset_ = threads_offset;
    }
}}    // namespace hpx::threads

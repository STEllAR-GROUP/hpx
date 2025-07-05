//  Copyright (c) 2007-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/affinity/affinity_data.hpp>
#include <hpx/hardware/timestamp.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/scheduler_state.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>
#include <hpx/timing/high_resolution_clock.hpp>
#include <hpx/topology/topology.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>

namespace hpx::threads {

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
    mask_type thread_pool_base::get_used_processing_units(
        std::size_t num_cores, bool full_cores) const
    {
        auto const& topo = create_topology();
        auto const sched = get_scheduler();

        auto used_processing_units = mask_type();
        threads::resize(used_processing_units,
            static_cast<std::size_t>(hardware_concurrency()));

        std::size_t const max_cores = get_os_thread_count();
        for (std::size_t thread_num = 0;
            thread_num != max_cores && num_cores != 0; ++thread_num)
        {
            if (sched->get_state(thread_num).load() <= hpx::state::suspended)
            {
                if (!full_cores)
                {
                    used_processing_units |= affinity_data_.get_pu_mask(
                        topo, thread_num + get_thread_offset());
                }
                else
                {
                    used_processing_units |= topo.get_core_affinity_mask(
                        thread_num + get_thread_offset());
                }
                --num_cores;
            }
        }

        return used_processing_units;
    }

    mask_type thread_pool_base::get_used_processing_units(bool full_cores) const
    {
        return get_used_processing_units(get_os_thread_count(), full_cores);
    }

    mask_type thread_pool_base::get_used_processing_unit(
        std::size_t thread_num, bool full_cores) const
    {
        auto const& topo = create_topology();
        if (!full_cores)
        {
            return affinity_data_.get_pu_mask(
                topo, thread_num + get_thread_offset());
        }
        return topo.get_core_affinity_mask(thread_num + get_thread_offset());
    }

    hwloc_bitmap_ptr thread_pool_base::get_numa_domain_bitmap() const
    {
        auto const& topo = create_topology();
        mask_type const used_processing_units = get_used_processing_units();
        return topo.cpuset_to_nodeset(used_processing_units);
    }

    std::int64_t thread_pool_base::get_thread_count_unknown(
        std::size_t num_thread, bool reset)
    {
        return get_thread_count(thread_schedule_state::unknown,
            thread_priority::default_, num_thread, reset);
    }

    std::int64_t thread_pool_base::get_thread_count_active(
        std::size_t num_thread, bool reset)
    {
        return get_thread_count(thread_schedule_state::active,
            thread_priority::default_, num_thread, reset);
    }

    std::int64_t thread_pool_base::get_thread_count_pending(
        std::size_t num_thread, bool reset)
    {
        return get_thread_count(thread_schedule_state::pending,
            thread_priority::default_, num_thread, reset);
    }

    std::int64_t thread_pool_base::get_thread_count_suspended(
        std::size_t num_thread, bool reset)
    {
        return get_thread_count(thread_schedule_state::suspended,
            thread_priority::default_, num_thread, reset);
    }

    std::int64_t thread_pool_base::get_thread_count_terminated(
        std::size_t num_thread, bool reset)
    {
        return get_thread_count(thread_schedule_state::terminated,
            thread_priority::default_, num_thread, reset);
    }

    std::int64_t thread_pool_base::get_thread_count_staged(
        std::size_t num_thread, bool reset)
    {
        return get_thread_count(thread_schedule_state::staged,
            thread_priority::default_, num_thread, reset);
    }

    std::size_t thread_pool_base::get_active_os_thread_count() const
    {
        std::size_t active_os_thread_count = 0;

        for (std::size_t thread_num = 0; thread_num < get_os_thread_count();
            ++thread_num)
        {
            if (get_scheduler()->get_state(thread_num).load() <=
                hpx::state::suspended)
            {
                ++active_os_thread_count;
            }
        }

        return active_os_thread_count;
    }

    ///////////////////////////////////////////////////////////////////////////
    void thread_pool_base::init_pool_time_scale()
    {
        // scale timestamps to nanoseconds
        std::uint64_t const base_timestamp = util::hardware::timestamp();
        std::uint64_t const base_time =
            hpx::chrono::high_resolution_clock::now();
        std::uint64_t curr_timestamp = util::hardware::timestamp();
        std::uint64_t curr_time = hpx::chrono::high_resolution_clock::now();

        while ((curr_time - base_time) <= 100000)
        {
            curr_timestamp = util::hardware::timestamp();
            curr_time = hpx::chrono::high_resolution_clock::now();
        }

        if (curr_timestamp - base_timestamp != 0)
        {
            timestamp_scale_ = static_cast<double>(curr_time - base_time) /
                static_cast<double>(curr_timestamp - base_timestamp);
        }
    }

    void thread_pool_base::init(
        std::size_t /* pool_threads */, std::size_t threads_offset)
    {
        thread_offset_ = threads_offset;
    }

    std::ostream& operator<<(
        std::ostream& os, thread_pool_base const& thread_pool)
    {
        auto const id = thread_pool.get_pool_id();
        os << id.name() << "(" << static_cast<std::uint64_t>(id.index()) << ")";

        return os;
    }
}    // namespace hpx::threads

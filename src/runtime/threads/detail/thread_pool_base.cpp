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
#include <hpx/runtime/threads/detail/thread_num_tss.hpp>
#include <hpx/runtime/threads/detail/thread_pool_base.hpp>
#include <hpx/runtime/threads/policies/callback_notifier.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/hardware/timestamp.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/util/thread_specific_ptr.hpp>

#include <boost/atomic.hpp>
#include <boost/system/system_error.hpp>

#include <cstddef>
#include <cstdint>

namespace hpx { namespace threads { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    thread_pool_base::thread_pool_base(
            threads::policies::callback_notifier& notifier,
            std::size_t index, std::string const& pool_name,
            policies::scheduler_mode m, std::size_t thread_offset)
      : id_(index, pool_name),
        used_processing_units_(),
        mode_(m),
        thread_offset_(thread_offset),
        timestamp_scale_(1.0),
        notifier_(notifier)
    {}

    ///////////////////////////////////////////////////////////////////////////
    mask_cref_type thread_pool_base::get_used_processing_units() const
    {
        return used_processing_units_;
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t thread_pool_base::get_worker_thread_num() const
    {
        return thread_num_tss_.get_worker_thread_num();
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

        auto const& rp = get_resource_partitioner();

        resize(used_processing_units_, threads::hardware_concurrency());
        for (std::size_t i = 0; i != pool_threads; ++i)
        {
            used_processing_units_ |= rp.get_pu_mask(threads_offset + i);
        }
    }

    bool thread_pool_base::run(
        std::unique_lock<compat::mutex>& l, std::size_t num_threads)
    {
        compat::barrier startup(num_threads + 1);
        bool ret = run(l, startup, num_threads);
        startup.wait();
        return ret;
    }
}}}


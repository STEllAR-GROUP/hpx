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
#include <hpx/runtime/threads/detail/thread_pool.hpp>
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
    thread_pool::thread_pool(
            std::size_t index, char const* pool_name,
            policies::scheduler_mode m, std::size_t thread_offset)
      : id_(index, pool_name),
        used_processing_units_(),
        mode_(m),
        thread_offset_(thread_offset),
        timestamp_scale_(1.0)
    {}

    ///////////////////////////////////////////////////////////////////////////
    mask_cref_type thread_pool::get_used_processing_units() const
    {
        return used_processing_units_;
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_state thread_pool::set_state(
        thread_id_type const& id, thread_state_enum new_state,
        thread_state_ex_enum new_state_ex, thread_priority priority,
        error_code& ec)
    {
        return detail::set_thread_state(id, new_state, //-V107
            new_state_ex, priority, get_worker_thread_num(), ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t thread_pool::get_worker_thread_num() const
    {
        return thread_num_tss_.get_worker_thread_num();
    }

    ///////////////////////////////////////////////////////////////////////////
    void thread_pool::init_pool_time_scale()
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
}}}


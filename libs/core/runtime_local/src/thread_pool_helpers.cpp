//  Copyright (c)      2017 Shoshana Jakobovits
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/modules/resource_partitioner.hpp>
#include <hpx/modules/threadmanager.hpp>
#include <hpx/runtime_local/runtime_local.hpp>
#include <hpx/runtime_local/thread_pool_helpers.hpp>
#include <hpx/topology/cpu_mask.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hpx { namespace resource {
    std::size_t get_num_thread_pools()
    {
        return get_partitioner().get_num_pools();
    }

    std::size_t get_num_threads()
    {
        return get_partitioner().get_num_threads();
    }

    std::size_t get_num_threads(std::string const& pool_name)
    {
        return get_partitioner().get_num_threads(pool_name);
    }

    std::size_t get_num_threads(std::size_t pool_index)
    {
        return get_partitioner().get_num_threads(pool_index);
    }

    std::size_t get_pool_index(std::string const& pool_name)
    {
        return get_partitioner().get_pool_index(pool_name);
    }

    std::string const& get_pool_name(std::size_t pool_index)
    {
        return get_partitioner().get_pool_name(pool_index);
    }

    threads::thread_pool_base& get_thread_pool(std::string const& pool_name)
    {
        return get_runtime().get_thread_manager().get_pool(pool_name);
    }

    threads::thread_pool_base& get_thread_pool(std::size_t pool_index)
    {
        return get_thread_pool(get_pool_name(pool_index));
    }

    bool pool_exists(std::string const& pool_name)
    {
        return get_runtime().get_thread_manager().pool_exists(pool_name);
    }

    bool pool_exists(std::size_t pool_index)
    {
        return get_runtime().get_thread_manager().pool_exists(pool_index);
    }
}}    // namespace hpx::resource

namespace hpx { namespace threads {
    std::int64_t get_thread_count(thread_schedule_state state)
    {
        return get_thread_manager().get_thread_count(state);
    }

    std::int64_t get_thread_count(
        thread_priority priority, thread_schedule_state state)
    {
        return get_thread_manager().get_thread_count(state, priority);
    }

    std::int64_t get_idle_core_count()
    {
        return get_thread_manager().get_idle_core_count();
    }

    mask_type get_idle_core_mask()
    {
        return get_thread_manager().get_idle_core_mask();
    }

    bool enumerate_threads(util::function_nonser<bool(thread_id_type)> const& f,
        thread_schedule_state state)
    {
        return get_thread_manager().enumerate_threads(f, state);
    }
}}    // namespace hpx::threads

//  Copyright (c)      2017 Shoshana Jakobovits
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/threading_base/network_background_callback.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>
#include <hpx/threading_base/thread_queue_init_parameters.hpp>

#include <cstddef>
#include <memory>
#include <string>

namespace hpx { namespace resource {
    class numa_domain;
    class core;
    class pu;

    class partitioner;

    namespace detail {
        class HPX_EXPORT partitioner;
        void HPX_EXPORT delete_partitioner();
    }    // namespace detail

    /// May be used anywhere in code and returns a reference to the single,
    /// global resource partitioner.
    HPX_EXPORT detail::partitioner& get_partitioner();

    /// Returns true if the resource partitioner has been initialized.
    /// Returns false otherwise.
    HPX_EXPORT bool is_partitioner_valid();

    /// This enumeration describes the modes available when creating a
    /// resource partitioner.
    enum partitioner_mode
    {
        /// Default mode.
        mode_default = 0,
        /// Allow processing units to be oversubscribed, i.e. multiple
        /// worker threads to share a single processing unit.
        mode_allow_oversubscription = 1,
        /// Allow worker threads to be added and removed from thread pools.
        mode_allow_dynamic_pools = 2
    };

    using scheduler_function =
        util::function_nonser<std::unique_ptr<hpx::threads::thread_pool_base>(
            hpx::threads::thread_pool_init_parameters,
            hpx::threads::policies::thread_queue_init_parameters)>;

    // Choose same names as in command-line options except with _ instead of
    // -.

    /// This enumeration lists the available scheduling policies (or
    /// schedulers) when creating thread pools.
    enum scheduling_policy
    {
        user_defined = -2,
        unspecified = -1,
        local = 0,
        local_priority_fifo = 1,
        local_priority_lifo = 2,
        static_ = 3,
        static_priority = 4,
        abp_priority_fifo = 5,
        abp_priority_lifo = 6,
        shared_priority = 7,
    };
}}    // namespace hpx::resource

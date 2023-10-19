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
#include <cstdint>
#include <memory>
#include <string>

namespace hpx::resource {

    class numa_domain;
    class core;
    class pu;

    class partitioner;

    namespace detail {

        class HPX_CORE_EXPORT partitioner;
        void HPX_CORE_EXPORT delete_partitioner();
    }    // namespace detail

    /// May be used anywhere in code and returns a reference to the single,
    /// global resource partitioner.
    HPX_CORE_EXPORT detail::partitioner& get_partitioner();

    /// Returns true if the resource partitioner has been initialized.
    /// Returns false otherwise.
    HPX_CORE_EXPORT bool is_partitioner_valid();

    /// This enumeration describes the modes available when creating a
    /// resource partitioner.
    enum class partitioner_mode : std::int8_t
    {
        /// Default mode.
        default_ = 0,

        /// Allow processing units to be oversubscribed, i.e. multiple
        /// worker threads to share a single processing unit.
        allow_oversubscription = 1,

        /// Allow worker threads to be added and removed from thread pools.
        allow_dynamic_pools = 2
    };

#define HPX_PARTITIONER_MODE_UNSCOPED_ENUM_DEPRECATION_MSG                     \
    "The unscoped partitioner_mode names are deprecated. Please use "          \
    "partitioner_mode::state instead."

    HPX_DEPRECATED_V(1, 9, HPX_PARTITIONER_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr partitioner_mode mode_default = partitioner_mode::default_;
    HPX_DEPRECATED_V(1, 9, HPX_PARTITIONER_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr partitioner_mode mode_allow_oversubscription =
        partitioner_mode::allow_oversubscription;
    HPX_DEPRECATED_V(1, 9, HPX_PARTITIONER_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr partitioner_mode mode_allow_dynamic_pools =
        partitioner_mode::allow_dynamic_pools;

#undef HPX_PARTITIONER_MODE_UNSCOPED_ENUM_DEPRECATION_MSG

    constexpr partitioner_mode operator&(
        partitioner_mode lhs, partitioner_mode rhs) noexcept
    {
        return static_cast<partitioner_mode>(
            static_cast<int>(lhs) & static_cast<int>(rhs));
    }

    constexpr bool as_bool(partitioner_mode val) noexcept
    {
        return static_cast<int>(val) != 0;
    }

    using scheduler_function =
        hpx::function<std::unique_ptr<hpx::threads::thread_pool_base>(
            hpx::threads::thread_pool_init_parameters,
            hpx::threads::policies::thread_queue_init_parameters)>;

    using background_work_function = hpx::function<bool(std::size_t)>;

    // Choose same names as in command-line options except with _ instead of
    // -.

    /// This enumeration lists the available scheduling policies (or
    /// schedulers) when creating thread pools.
    enum class scheduling_policy : std::int8_t
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
        local_workrequesting_fifo = 8,
        local_workrequesting_lifo = 9,
    };

#define HPX_SCHEDULING_POLICY_UNSCOPED_ENUM_DEPRECATION_MSG                    \
    "The unscoped scheduling_policy names are deprecated. Please use "         \
    "scheduling_policy::state instead."

    HPX_DEPRECATED_V(1, 9, HPX_SCHEDULING_POLICY_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduling_policy user_defined =
        scheduling_policy::user_defined;
    HPX_DEPRECATED_V(1, 9, HPX_SCHEDULING_POLICY_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduling_policy unspecified =
        scheduling_policy::unspecified;
    HPX_DEPRECATED_V(1, 9, HPX_SCHEDULING_POLICY_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduling_policy local = scheduling_policy::local;
    HPX_DEPRECATED_V(1, 9, HPX_SCHEDULING_POLICY_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduling_policy local_priority_fifo =
        scheduling_policy::local_priority_fifo;
    HPX_DEPRECATED_V(1, 9, HPX_SCHEDULING_POLICY_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduling_policy local_priority_lifo =
        scheduling_policy::local_priority_lifo;
    HPX_DEPRECATED_V(1, 9, HPX_SCHEDULING_POLICY_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduling_policy static_ = scheduling_policy::static_;
    HPX_DEPRECATED_V(1, 9, HPX_SCHEDULING_POLICY_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduling_policy static_priority =
        scheduling_policy::static_priority;
    HPX_DEPRECATED_V(1, 9, HPX_SCHEDULING_POLICY_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduling_policy abp_priority_fifo =
        scheduling_policy::abp_priority_fifo;
    HPX_DEPRECATED_V(1, 9, HPX_SCHEDULING_POLICY_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduling_policy abp_priority_lifo =
        scheduling_policy::abp_priority_lifo;
    HPX_DEPRECATED_V(1, 9, HPX_SCHEDULING_POLICY_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduling_policy shared_priority =
        scheduling_policy::shared_priority;

#undef HPX_SCHEDULING_POLICY_UNSCOPED_ENUM_DEPRECATION_MSG
}    // namespace hpx::resource

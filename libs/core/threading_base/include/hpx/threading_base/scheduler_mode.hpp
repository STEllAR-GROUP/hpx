//  Copyright (c) 2015-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstdint>

namespace hpx { namespace threads { namespace policies {

    /// This enumeration describes the possible modes of a scheduler.
    enum class scheduler_mode : std::uint32_t
    {
        /// As the name suggests, this option can be used to disable all other
        /// options.
        nothing_special = 0x0000,
        /// The scheduler will periodically call a provided callback function
        /// from a special HPX thread to enable performing background-work, for
        /// instance driving networking progress or garbage-collect AGAS.
        do_background_work = 0x0001,
        /// The kernel priority of the os-thread driving the scheduler will be
        /// reduced below normal.
        reduce_thread_priority = 0x0002,
        /// The scheduler will wait for some unspecified amount of time before
        /// exiting the scheduling loop while being terminated to make sure no
        /// other work is being scheduled during processing the shutdown
        /// request.
        delay_exit = 0x0004,
        /// Some schedulers have the capability to act as 'embedded'
        /// schedulers. In this case it needs to periodically invoke a provided
        /// callback into the outer scheduler more frequently than normal. This
        /// option enables this behavior.
        fast_idle_mode = 0x0008,
        /// This option allows for the scheduler to dynamically increase and
        /// reduce the number of processing units it runs on. Setting this value
        /// not succeed for schedulers that do not support this functionality.
        enable_elasticity = 0x0010,
        /// This option allows schedulers that support work thread/stealing to
        /// enable/disable it
        enable_stealing = 0x0020,
        /// This option allows schedulersthat support it to disallow stealing
        /// between numa domains
        enable_stealing_numa = 0x0040,
        /// This option tells schedulersthat support it to add tasks round
        /// robin to queues on each core
        assign_work_round_robin = 0x0080,
        /// This option tells schedulers that support it to add tasks round to
        /// the same core/queue that the parent task is running on
        assign_work_thread_parent = 0x0100,
        /// This option tells schedulers that support it to always (try to)
        /// steal high priority tasks from other queues before finishing their
        /// own lower priority tasks
        steal_high_priority_first = 0x0200,
        /// This option tells schedulers that support it to steal tasks only
        /// when their local queues are empty
        steal_after_local = 0x0400,
        /// This option allows for certain schedulers to explicitly disable
        /// exponential idle-back off
        enable_idle_backoff = 0x0800,

        // clang-format off
        /// This option represents the default mode.
        default_ =
            do_background_work |
            reduce_thread_priority |
            delay_exit |
            enable_stealing |
            enable_stealing_numa |
            assign_work_round_robin |
            steal_after_local |
            enable_idle_backoff,
        /// This enables all available options.
        all_flags =
            do_background_work |
            reduce_thread_priority |
            delay_exit |
            fast_idle_mode |
            enable_elasticity |
            enable_stealing |
            enable_stealing_numa |
            assign_work_round_robin |
            assign_work_thread_parent |
            steal_high_priority_first |
            steal_after_local |
            enable_idle_backoff
        // clang-format on
    };

    inline constexpr scheduler_mode operator|(
        scheduler_mode lhs, scheduler_mode rhs) noexcept
    {
        return static_cast<scheduler_mode>(
            static_cast<std::uint32_t>(lhs) | static_cast<std::uint32_t>(rhs));
    }

    inline constexpr scheduler_mode operator|(
        std::uint32_t lhs, scheduler_mode rhs) noexcept
    {
        return static_cast<scheduler_mode>(
            lhs | static_cast<std::uint32_t>(rhs));
    }

    inline constexpr scheduler_mode operator|(
        scheduler_mode lhs, std::uint32_t rhs) noexcept
    {
        return static_cast<scheduler_mode>(
            static_cast<std::uint32_t>(lhs) | rhs);
    }

    inline constexpr std::uint32_t operator&(
        scheduler_mode lhs, scheduler_mode rhs) noexcept
    {
        return static_cast<std::uint32_t>(lhs) &
            static_cast<std::uint32_t>(rhs);
    }

    inline constexpr std::uint32_t operator&(
        std::uint32_t lhs, scheduler_mode rhs) noexcept
    {
        return lhs & static_cast<std::uint32_t>(rhs);
    }

    inline constexpr std::uint32_t operator&(
        scheduler_mode lhs, std::uint32_t rhs) noexcept
    {
        return static_cast<std::uint32_t>(lhs) & rhs;
    }

    inline constexpr std::uint32_t operator~(scheduler_mode mode) noexcept
    {
        return ~static_cast<std::uint32_t>(mode);
    }

#define HPX_SCHEDULER_MODE_UNSCOPED_ENUM_DEPRECATION_MSG                       \
    "The unscoped scheduler_mode names are deprecated. Please use "            \
    "scheduler_mode::state instead."

    HPX_DEPRECATED_V(1, 8, HPX_SCHEDULER_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduler_mode nothing_special =
        scheduler_mode::nothing_special;
    HPX_DEPRECATED_V(1, 8, HPX_SCHEDULER_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduler_mode do_background_work =
        scheduler_mode::do_background_work;
    HPX_DEPRECATED_V(1, 8, HPX_SCHEDULER_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduler_mode reduce_thread_priority =
        scheduler_mode::reduce_thread_priority;
    HPX_DEPRECATED_V(1, 8, HPX_SCHEDULER_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduler_mode delay_exit = scheduler_mode::delay_exit;
    HPX_DEPRECATED_V(1, 8, HPX_SCHEDULER_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduler_mode fast_idle_mode =
        scheduler_mode::fast_idle_mode;
    HPX_DEPRECATED_V(1, 8, HPX_SCHEDULER_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduler_mode enable_elasticity =
        scheduler_mode::enable_elasticity;
    HPX_DEPRECATED_V(1, 8, HPX_SCHEDULER_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduler_mode enable_stealing =
        scheduler_mode::enable_stealing;
    HPX_DEPRECATED_V(1, 8, HPX_SCHEDULER_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduler_mode enable_stealing_numa =
        scheduler_mode::enable_stealing_numa;
    HPX_DEPRECATED_V(1, 8, HPX_SCHEDULER_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduler_mode assign_work_round_robin =
        scheduler_mode::assign_work_round_robin;
    HPX_DEPRECATED_V(1, 8, HPX_SCHEDULER_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduler_mode assign_work_thread_parent =
        scheduler_mode::assign_work_thread_parent;
    HPX_DEPRECATED_V(1, 8, HPX_SCHEDULER_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduler_mode steal_high_priority_first =
        scheduler_mode::steal_high_priority_first;
    HPX_DEPRECATED_V(1, 8, HPX_SCHEDULER_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduler_mode steal_after_local =
        scheduler_mode::steal_after_local;
    HPX_DEPRECATED_V(1, 8, HPX_SCHEDULER_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduler_mode enable_idle_backoff =
        scheduler_mode::enable_idle_backoff;
    HPX_DEPRECATED_V(1, 8, HPX_SCHEDULER_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduler_mode default_mode = scheduler_mode::default_;
    HPX_DEPRECATED_V(1, 8, HPX_SCHEDULER_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr scheduler_mode all_flags = scheduler_mode::all_flags;

#undef HPX_SCHEDULER_MODE_UNSCOPED_ENUM_DEPRECATION_MSG
}}}    // namespace hpx::threads::policies

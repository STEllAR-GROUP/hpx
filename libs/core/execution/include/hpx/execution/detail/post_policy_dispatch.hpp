//  Copyright (c) 2017-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/execution/detail/sync_launch_policy_dispatch.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/threading_base/thread_description.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/threading_base/thread_num_tss.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace hpx::detail {

    ////////////////////////////////////////////////////////////////////////////
    // forward declaration
    template <typename Policy>
    struct post_policy_dispatch;

    template <>
    struct post_policy_dispatch<launch::async_policy>
    {
        template <typename Policy, typename F, typename... Ts>
        static void call(Policy const& policy,
            hpx::threads::thread_description const& desc,
            threads::thread_pool_base* pool, F&& f, Ts&&... ts)
        {
            threads::thread_init_data data(
                threads::make_thread_function_nullary(hpx::util::deferred_call(
                    HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...)),
                desc, policy.priority(), policy.hint(), policy.stacksize(),
                threads::thread_schedule_state::pending);

            threads::register_work(data, pool);
        }

        template <typename Policy, typename F, typename... Ts>
        static void call(Policy const& policy,
            hpx::threads::thread_description const& desc, F&& f, Ts&&... ts)
        {
            call(policy, desc, threads::detail::get_self_or_default_pool(),
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }
    };

    template <>
    struct post_policy_dispatch<launch::fork_policy>
    {
        template <typename Policy, typename F, typename... Ts>
        static void call(Policy const& policy,
            hpx::threads::thread_description const& desc,
            threads::thread_pool_base* pool, F&& f, Ts&&... ts)
        {
            threads::thread_init_data data(
                threads::make_thread_function_nullary(hpx::util::deferred_call(
                    HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...)),
                desc, policy.priority(),
                threads::thread_schedule_hint(
                    static_cast<std::int16_t>(get_worker_thread_num())),
                policy.stacksize(),
                threads::thread_schedule_state::pending_do_not_schedule, true);

            threads::thread_id_ref_type tid =
                threads::register_thread(data, pool);
            threads::thread_id_type tid_self = threads::get_self_id();

            // make sure this thread is executed last
            if (tid && tid_self &&
                get_thread_id_data(tid)->get_scheduler_base() ==
                    get_thread_id_data(tid_self)->get_scheduler_base())
            {
                // yield_to(tid)
                hpx::this_thread::suspend(
                    threads::thread_schedule_state::pending, tid.noref(),
                    "post_policy_dispatch(suspend)");
            }
        }

        template <typename Policy, typename F, typename... Ts>
        static void call(Policy const& policy,
            hpx::threads::thread_description const& desc, F&& f, Ts&&... ts)
        {
            call(policy, desc, threads::detail::get_self_or_default_pool(),
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }
    };

    template <>
    struct post_policy_dispatch<launch::sync_policy>
    {
        template <typename Policy, typename F, typename... Ts>
        static void call(Policy const& policy,
            hpx::threads::thread_description const&, threads::thread_pool_base*,
            F&& f, Ts&&... ts)
        {
            hpx::detail::sync_launch_policy_dispatch<launch::sync_policy>::call(
                policy, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        template <typename Policy, typename F, typename... Ts>
        static void call(Policy const& policy,
            hpx::threads::thread_description const&, F&& f, Ts&&... ts)
        {
            hpx::detail::sync_launch_policy_dispatch<launch::sync_policy>::call(
                policy, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }
    };

    template <>
    struct post_policy_dispatch<launch::deferred_policy>
    {
        template <typename Policy, typename F, typename... Ts>
        static void call(Policy const& policy,
            hpx::threads::thread_description const&, threads::thread_pool_base*,
            F&& f, Ts&&... ts)
        {
            hpx::detail::sync_launch_policy_dispatch<
                launch::deferred_policy>::call(policy, HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
        }

        template <typename Policy, typename F, typename... Ts>
        static void call(Policy const& policy,
            hpx::threads::thread_description const&, F&& f, Ts&&... ts)
        {
            hpx::detail::sync_launch_policy_dispatch<
                launch::deferred_policy>::call(policy, HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
        }
    };

    template <typename Policy>
    struct post_policy_dispatch
    {
        template <typename F, typename... Ts>
        static void call(Policy const& policy,
            hpx::threads::thread_description const& desc,
            threads::thread_pool_base* pool, F&& f, Ts&&... ts)
        {
            if (policy == launch::sync)
            {
                post_policy_dispatch<launch::sync_policy>::call(policy, desc,
                    pool, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
            }
            else if (policy == launch::deferred)
            {
                // execute synchronously
                post_policy_dispatch<launch::deferred_policy>::call(policy,
                    desc, pool, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
            }
            else if (policy == launch::fork)
            {
                post_policy_dispatch<launch::fork_policy>::call(policy, desc,
                    pool, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
            }
            else
            {
                post_policy_dispatch<launch::async_policy>::call(policy, desc,
                    pool, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
            }
        }

        template <typename F, typename... Ts>
        static void call(Policy const& policy,
            hpx::threads::thread_description const& desc, F&& f, Ts&&... ts)
        {
            call(policy, desc, threads::detail::get_self_or_default_pool(),
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }
    };
}    // namespace hpx::detail

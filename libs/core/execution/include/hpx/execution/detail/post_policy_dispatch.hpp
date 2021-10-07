//  Copyright (c) 2017-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/execution/detail/async_launch_policy_dispatch.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/threading_base/thread_description.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/threading_base/thread_num_tss.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace execution { namespace detail {

    ////////////////////////////////////////////////////////////////////////////
    // forward declaration
    template <typename Policy>
    struct post_policy_dispatch;

    template <>
    struct post_policy_dispatch<launch::fork_policy>
    {
        template <typename F, typename... Ts>
        static void call(launch::fork_policy const& policy,
            hpx::util::thread_description const& desc,
            threads::thread_pool_base* pool, F&& f, Ts&&... ts)
        {
            threads::thread_init_data data(
                threads::make_thread_function_nullary(hpx::util::deferred_call(
                    std::forward<F>(f), std::forward<Ts>(ts)...)),
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

        template <typename F, typename... Ts>
        static void call(launch::fork_policy const& policy,
            hpx::util::thread_description const& desc, F&& f, Ts&&... ts)
        {
            call(policy, desc, threads::detail::get_self_or_default_pool(),
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }
    };

    template <>
    struct post_policy_dispatch<launch::sync_policy>
    {
        template <typename F, typename... Ts>
        static void call(launch::sync_policy const&,
            hpx::util::thread_description const& /* desc */,
            threads::thread_pool_base* /* pool */, F&& f, Ts&&... ts)
        {
            hpx::detail::call_sync(std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename F, typename... Ts>
        static void call(launch::sync_policy const& /* policy */,
            hpx::util::thread_description const& /* desc */, F&& f,
            Ts&&... ts) noexcept
        {
            hpx::detail::call_sync(std::forward<F>(f), std::forward<Ts>(ts)...);
        }
    };

    template <typename Policy>
    struct post_policy_dispatch
    {
        template <typename F, typename... Ts>
        static void call(Policy const& policy,
            hpx::util::thread_description const& desc,
            threads::thread_pool_base* pool, F&& f, Ts&&... ts)
        {
            if (policy == launch::sync)
            {
                hpx::detail::call_sync(
                    std::forward<F>(f), std::forward<Ts>(ts)...);
                return;
            }
            else if (policy == launch::fork)
            {
                auto fork_policy = launch::fork_policy(
                    policy.priority(), policy.stacksize(), policy.hint());

                post_policy_dispatch<launch::fork_policy>::call(fork_policy,
                    desc, pool, std::forward<F>(f), std::forward<Ts>(ts)...);
                return;
            }

            threads::thread_init_data data(
                threads::make_thread_function_nullary(hpx::util::deferred_call(
                    std::forward<F>(f), std::forward<Ts>(ts)...)),
                desc, policy.priority(), policy.hint(), policy.stacksize(),
                threads::thread_schedule_state::pending);

            threads::register_work(data, pool);
        }

        template <typename F, typename... Ts>
        static void call(Policy const& policy,
            hpx::util::thread_description const& desc, F&& f, Ts&&... ts)
        {
            if (policy == launch::sync)
            {
                hpx::detail::call_sync(
                    std::forward<F>(f), std::forward<Ts>(ts)...);
                return;
            }
            else if (policy == launch::fork)
            {
                auto fork_policy = launch::fork_policy(
                    policy.priority(), policy.stacksize(), policy.hint());

                post_policy_dispatch<launch::fork_policy>::call(fork_policy,
                    desc, std::forward<F>(f), std::forward<Ts>(ts)...);
                return;
            }

            threads::thread_init_data data(
                threads::make_thread_function_nullary(hpx::util::deferred_call(
                    std::forward<F>(f), std::forward<Ts>(ts)...)),
                desc, policy.priority(), policy.hint(), policy.stacksize(),
                threads::thread_schedule_state::pending);

            threads::register_work(data);
        }
    };
}}}}    // namespace hpx::parallel::execution::detail

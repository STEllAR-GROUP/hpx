//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(                                                                  \
    HPX_PARALLEL_EXECUTION_DETAIL_POST_POLICY_DISPATCH_DEC_05_2017_0234PM)
#define HPX_PARALLEL_EXECUTION_DETAIL_POST_POLICY_DISPATCH_DEC_05_2017_0234PM

#include <hpx/config.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/runtime/get_worker_thread_num.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/thread_pool_base.hpp>

#include <cstdint>
#include <utility>

namespace hpx { namespace parallel { namespace execution { namespace detail {

    ////////////////////////////////////////////////////////////////////////////
    template <typename Policy>
    struct post_policy_dispatch
    {
        template <typename F, typename... Ts>
        static void call(Policy const& policy,
            hpx::util::thread_description const& desc,
            threads::thread_pool_base* pool, threads::thread_schedule_hint hint,
            F&& f, Ts&&... ts)
        {
            threads::register_thread_nullary(pool,
                hpx::util::deferred_call(
                    std::forward<F>(f), std::forward<Ts>(ts)...),
                desc, threads::pending, false, policy.priority(), hint);
        }

        template <typename F, typename... Ts>
        static void call(Policy const& policy,
            hpx::util::thread_description const& desc, F&& f, Ts&&... ts)
        {
            threads::register_thread_nullary(
                hpx::util::deferred_call(
                    std::forward<F>(f), std::forward<Ts>(ts)...),
                desc, threads::pending, false, policy.priority());
        }
    };

    template <>
    struct post_policy_dispatch<launch::fork_policy>
    {
        template <typename F, typename... Ts>
        static void call(launch::fork_policy const& policy,
            hpx::util::thread_description const& desc,
            threads::thread_pool_base* pool, threads::thread_schedule_hint hint,
            F&& f, Ts&&... ts)
        {
            hint.mode = threads::thread_schedule_hint_mode_thread;
            hint.hint = static_cast<std::int16_t>(get_worker_thread_num());
            threads::thread_id_type tid = threads::register_thread_nullary(pool,
                hpx::util::deferred_call(
                    std::forward<F>(f), std::forward<Ts>(ts)...),
                desc, threads::pending_do_not_schedule, true, policy.priority(),
                hint, threads::thread_stacksize_current);
            threads::thread_id_type tid_self = threads::get_self_id();

            // make sure this thread is executed last
            if (tid && tid_self &&
                get_thread_id_data(tid)->get_scheduler_base() ==
                    get_thread_id_data(tid_self)->get_scheduler_base())
            {
                // yield_to(tid)
                hpx::this_thread::suspend(threads::pending, tid,
                    "hpx::parallel::execution::parallel_executor::post");
            }
        }

        template <typename F, typename... Ts>
        static void call(launch::fork_policy const& policy,
            hpx::util::thread_description const& desc, F&& f, Ts&&... ts)
        {
            call(policy, desc, threads::detail::get_self_or_default_pool(),
                threads::thread_schedule_hint{}, std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }
    };
}}}}    // namespace hpx::parallel::execution::detail

#endif

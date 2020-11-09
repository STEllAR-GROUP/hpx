//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/detail/invoke.hpp>
#include <hpx/functional/traits/is_action.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/futures_factory.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace hpx { namespace detail {
    // dispatch point used for launch_policy implementations
    template <typename Action, typename Enable = void>
    struct async_launch_policy_dispatch;

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    HPX_FORCEINLINE lcos::future<typename util::detail::invoke_deferred_result<
        typename std::decay<F>::type, Ts...>::type>
    call_sync(std::false_type, F&& f, Ts... vs)
    {
        using R = typename util::detail::invoke_deferred_result<
            typename std::decay<F>::type, Ts...>::type;
        try
        {
            return lcos::make_ready_future<R>(
                HPX_INVOKE(std::forward<F>(f), std::move(vs)...));
        }
        catch (...)
        {
            return lcos::make_exceptional_future<R>(std::current_exception());
        }
    }

    template <typename F, typename... Ts>
    HPX_FORCEINLINE lcos::future<typename util::detail::invoke_deferred_result<
        typename std::decay<F>::type, Ts...>::type>
    call_sync(std::true_type, F&& f, Ts... vs)
    {
        try
        {
            HPX_INVOKE(std::forward<F>(f), std::move(vs)...);
            return lcos::make_ready_future();
        }
        catch (...)
        {
            return lcos::make_exceptional_future<void>(
                std::current_exception());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct async_launch_policy_dispatch<Action,
        typename std::enable_if<!traits::is_action<Action>::value>::type>
    {
        template <typename F, typename... Ts>
        HPX_FORCEINLINE static typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            hpx::future<typename util::detail::invoke_deferred_result<F,
                Ts...>::type>>::type
        call(launch policy, threads::thread_pool_base* pool,
            threads::thread_priority priority,
            threads::thread_stacksize stacksize,
            threads::thread_schedule_hint hint, F&& f, Ts&&... ts)
        {
            typedef
                typename util::detail::invoke_deferred_result<F, Ts...>::type
                    result_type;

            if (policy == launch::sync)
            {
                using is_void = typename std::is_void<result_type>::type;
                return detail::call_sync(
                    is_void{}, std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            lcos::local::futures_factory<result_type()> p(util::deferred_call(
                std::forward<F>(f), std::forward<Ts>(ts)...));
            if (hpx::detail::has_async_policy(policy))
            {
                threads::thread_id_type tid =
                    p.apply(pool, "async_launch_policy_dispatch", policy,
                        priority, stacksize, hint);
                if (tid && policy == launch::fork)
                {
                    // make sure this thread is executed last
                    // yield_to
                    hpx::this_thread::suspend(
                        threads::thread_schedule_state::pending, tid,
                        "async_launch_policy_dispatch<launch>");
                }
            }
            return p.get_future();
        }

        template <typename F, typename... Ts>
        HPX_FORCEINLINE static typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            hpx::future<typename util::detail::invoke_deferred_result<F,
                Ts...>::type>>::type
        call(launch policy, threads::thread_priority priority,
            threads::thread_stacksize stacksize,
            threads::thread_schedule_hint hint, F&& f, Ts&&... ts)
        {
            return call(policy, threads::detail::get_self_or_default_pool(),
                priority, stacksize, hint, std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }

        template <typename F, typename... Ts>
        HPX_FORCEINLINE static typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            hpx::future<typename util::detail::invoke_deferred_result<F,
                Ts...>::type>>::type
        call(launch policy, F&& f, Ts&&... ts)
        {
            return call(policy, threads::detail::get_self_or_default_pool(),
                policy.priority(), threads::thread_stacksize::default_,
                threads::thread_schedule_hint{}, std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }

        template <typename F, typename... Ts>
        HPX_FORCEINLINE static typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            hpx::future<typename util::detail::invoke_deferred_result<F,
                Ts...>::type>>::type
        call(hpx::detail::sync_policy, F&& f, Ts&&... ts)
        {
            typedef
                typename util::detail::invoke_deferred_result<F, Ts...>::type
                    result_type;

            using is_void = typename std::is_void<result_type>::type;
            return detail::call_sync(
                is_void{}, std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename F, typename... Ts>
        HPX_FORCEINLINE static typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            hpx::future<typename util::detail::invoke_deferred_result<F,
                Ts...>::type>>::type
        call(hpx::detail::async_policy policy, threads::thread_pool_base* pool,
            threads::thread_priority priority,
            threads::thread_stacksize stacksize,
            threads::thread_schedule_hint hint, F&& f, Ts&&... ts)
        {
            HPX_ASSERT(pool);
            typedef
                typename util::detail::invoke_deferred_result<F, Ts...>::type
                    result_type;

            lcos::local::futures_factory<result_type()> p(util::deferred_call(
                std::forward<F>(f), std::forward<Ts>(ts)...));

            p.apply(pool, "async_launch_policy_dispatch::call", policy,
                priority, stacksize, hint);
            return p.get_future();
        }

        template <typename F, typename... Ts>
        HPX_FORCEINLINE static typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            hpx::future<typename util::detail::invoke_deferred_result<F,
                Ts...>::type>>::type
        call(hpx::detail::async_policy policy, F&& f, Ts&&... ts)
        {
            return call(policy, threads::detail::get_self_or_default_pool(),
                threads::thread_priority::default_,
                threads::thread_stacksize::default_,
                threads::thread_schedule_hint{}, std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }

        template <typename F, typename... Ts>
        HPX_FORCEINLINE static typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            hpx::future<typename util::detail::invoke_deferred_result<F,
                Ts...>::type>>::type
        call(hpx::detail::fork_policy policy, threads::thread_pool_base* pool,
            threads::thread_priority priority,
            threads::thread_stacksize stacksize,
            threads::thread_schedule_hint hint, F&& f, Ts&&... ts)
        {
            HPX_ASSERT(pool);
            typedef
                typename util::detail::invoke_deferred_result<F, Ts...>::type
                    result_type;

            lcos::local::futures_factory<result_type()> p(util::deferred_call(
                std::forward<F>(f), std::forward<Ts>(ts)...));

            // make sure this thread is executed last
            threads::thread_id_type tid =
                p.apply(pool, "async_launch_policy_dispatch::call", policy,
                    priority, stacksize, hint);
            threads::thread_id_type tid_self = threads::get_self_id();
            if (tid && tid_self &&
                get_thread_id_data(tid)->get_scheduler_base() ==
                    get_thread_id_data(tid_self)->get_scheduler_base())
            {
                // yield_to
                hpx::this_thread::suspend(
                    threads::thread_schedule_state::pending, tid,
                    "async_launch_policy_dispatch<fork>");
            }
            return p.get_future();
        }

        template <typename F, typename... Ts>
        HPX_FORCEINLINE static typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            hpx::future<typename util::detail::invoke_deferred_result<F,
                Ts...>::type>>::type
        call(hpx::detail::fork_policy policy, F&& f, Ts&&... ts)
        {
            return call(policy, threads::detail::get_self_or_default_pool(),
                threads::thread_priority::default_,
                threads::thread_stacksize::default_,
                threads::thread_schedule_hint{}, std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }

        template <typename F, typename... Ts>
        HPX_FORCEINLINE static typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            hpx::future<typename util::detail::invoke_deferred_result<F,
                Ts...>::type>>::type
        call(hpx::detail::deferred_policy, F&& f, Ts&&... ts)
        {
            typedef
                typename util::detail::invoke_deferred_result<F, Ts...>::type
                    result_type;

            lcos::local::futures_factory<result_type()> p(util::deferred_call(
                std::forward<F>(f), std::forward<Ts>(ts)...));

            return p.get_future();
        }
    };
}}    // namespace hpx::detail

//  Copyright (c) 2017-2023 Hartmut Kaiser
//  Copyright (c) 2017 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/traits/executor_traits.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/timed_execution/timed_execution_fwd.hpp>
#include <hpx/timing/steady_clock.hpp>
#include <hpx/type_support/detail/wrap_int.hpp>

#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::parallel::execution::detail {

    // customization point for NonBlockingOneWayExecutor interface
    // post_at(), post_after()

    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////
    template <typename Executor>
    struct timed_async_execute_fn_helper<Executor,
        std::enable_if_t<hpx::traits::is_two_way_executor_v<Executor>>>
    {
        template <typename TwoWayExecutor, typename F, typename... Ts>
        HPX_FORCEINLINE static auto call(TwoWayExecutor&& exec,
            hpx::chrono::steady_time_point const& abs_time, F&& f, Ts&&... ts)
            -> decltype(execution::async_execute(
                timed_executor<TwoWayExecutor&>(exec, abs_time),
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...))
        {
            return execution::async_execute(
                timed_executor<TwoWayExecutor&>(exec, abs_time),
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        template <typename TwoWayExecutor, typename F, typename... Ts>
        HPX_FORCEINLINE static auto call(TwoWayExecutor&& exec,
            hpx::chrono::steady_duration const& rel_time, F&& f, Ts&&... ts)
            -> decltype(execution::async_execute(
                timed_executor<TwoWayExecutor&>(exec, rel_time),
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...))
        {
            return execution::async_execute(
                timed_executor<TwoWayExecutor&>(exec, rel_time),
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        template <typename TwoWayExecutor, typename F, typename... Ts>
        struct result
        {
            using type = decltype(call(std::declval<TwoWayExecutor>(),
                std::declval<hpx::chrono::steady_time_point const&>(),
                std::declval<F>(), std::declval<Ts>()...));
        };
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename Executor>
    struct timed_post_fn_helper<Executor,
        std::enable_if_t<hpx::traits::is_one_way_executor_v<Executor> ||
            hpx::traits::is_two_way_executor_v<Executor> ||
            hpx::traits::is_never_blocking_one_way_executor_v<Executor>>>
    {
        template <typename NonBlockingOneWayExecutor, typename F,
            typename... Ts>
        HPX_FORCEINLINE static auto call(NonBlockingOneWayExecutor&& exec,
            hpx::chrono::steady_time_point const& abs_time, F&& f, Ts&&... ts)
            -> decltype(execution::post(
                timed_executor<NonBlockingOneWayExecutor&>(exec, abs_time),
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...))
        {
            return execution::post(
                timed_executor<NonBlockingOneWayExecutor&>(exec, abs_time),
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        template <typename NonBlockingOneWayExecutor, typename F,
            typename... Ts>
        HPX_FORCEINLINE static auto call(NonBlockingOneWayExecutor&& exec,
            hpx::chrono::steady_duration const& rel_time, F&& f, Ts&&... ts)
            -> decltype(execution::post(
                timed_executor<NonBlockingOneWayExecutor&>(exec, rel_time),
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...))
        {
            return execution::post(
                timed_executor<NonBlockingOneWayExecutor&>(exec, rel_time),
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        template <typename NonBlockingOneWayExecutor, typename F,
            typename... Ts>
        struct result
        {
            using type =
                decltype(call(std::declval<NonBlockingOneWayExecutor>(),
                    std::declval<hpx::chrono::steady_time_point const&>(),
                    std::declval<F>(), std::declval<Ts>()...));
        };
    };
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    // customization points for OneWayExecutor interface
    // sync_execute_at(), sync_execute_after()

    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////
    template <typename Executor>
    struct timed_sync_execute_fn_helper<Executor,
        std::enable_if_t<hpx::traits::is_one_way_executor_v<Executor> ||
            hpx::traits::is_two_way_executor_v<Executor>>>
    {
        template <typename OneWayExecutor, typename F, typename... Ts>
        HPX_FORCEINLINE static auto call(OneWayExecutor&& exec,
            hpx::chrono::steady_time_point const& abs_time, F&& f, Ts&&... ts)
            -> decltype(execution::sync_execute(
                timed_executor<OneWayExecutor&>(exec, abs_time),
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...))
        {
            return execution::sync_execute(
                timed_executor<OneWayExecutor&>(exec, abs_time),
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        template <typename OneWayExecutor, typename F, typename... Ts>
        HPX_FORCEINLINE static auto call(OneWayExecutor&& exec,
            hpx::chrono::steady_duration const& rel_time, F&& f, Ts&&... ts)
            -> decltype(execution::sync_execute(
                timed_executor<OneWayExecutor&>(exec, rel_time),
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...))
        {
            return execution::sync_execute(
                timed_executor<OneWayExecutor&>(exec, rel_time),
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        template <typename OneWayExecutor, typename F, typename... Ts>
        struct result
        {
            using type = decltype(call(std::declval<OneWayExecutor>(),
                std::declval<hpx::chrono::steady_time_point const&>(),
                std::declval<F>(), std::declval<Ts>()...));
        };
    };

    /// \endcond
}    // namespace hpx::parallel::execution::detail

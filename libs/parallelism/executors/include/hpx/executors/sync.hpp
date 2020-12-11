//  Copyright (c) 2007-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/sync.hpp>
#include <hpx/execution/detail/sync_launch_policy_dispatch.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/executors/parallel_executor.hpp>
#include <hpx/functional/deferred_call.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace detail {
    template <typename Func, typename Enable = void>
    struct sync_dispatch_launch_policy_helper;

    template <typename Func>
    struct sync_dispatch_launch_policy_helper<Func,
        typename std::enable_if<!traits::is_action<Func>::value>::type>
    {
        template <typename Policy_, typename F, typename... Ts>
        HPX_FORCEINLINE static auto call(
            Policy_&& launch_policy, F&& f, Ts&&... ts)
            -> decltype(
                sync_launch_policy_dispatch<typename std::decay<F>::type>::call(
                    std::forward<Policy_>(launch_policy), std::forward<F>(f),
                    std::forward<Ts>(ts)...))
        {
            return sync_launch_policy_dispatch<typename std::decay<F>::type>::
                call(std::forward<Policy_>(launch_policy), std::forward<F>(f),
                    std::forward<Ts>(ts)...);
        }
    };

    template <typename Policy>
    struct sync_dispatch<Policy,
        typename std::enable_if<traits::is_launch_policy<Policy>::value>::type>
    {
        template <typename Policy_, typename F, typename... Ts>
        HPX_FORCEINLINE static auto call(
            Policy_&& launch_policy, F&& f, Ts&&... ts)
            -> decltype(sync_dispatch_launch_policy_helper<typename std::decay<
                    F>::type>::call(std::forward<Policy_>(launch_policy),
                std::forward<F>(f), std::forward<Ts>(ts)...))
        {
            return sync_dispatch_launch_policy_helper<typename std::decay<
                F>::type>::call(std::forward<Policy_>(launch_policy),
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }
    };

    // Launch the given function or function object synchronously. This exists
    // mostly for symmetry with hpx::async.
    template <typename Func, typename Enable>
    struct sync_dispatch
    {
        template <typename F, typename... Ts>
        HPX_FORCEINLINE static typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            typename util::detail::invoke_deferred_result<F, Ts...>::type>::type
        call(F&& f, Ts&&... ts)
        {
            execution::parallel_executor exec;
            return parallel::execution::sync_execute(
                exec, std::forward<F>(f), std::forward<Ts>(ts)...);
        }
    };

    // The overload for hpx::sync taking an executor simply forwards to the
    // corresponding executor customization point.
    //
    // parallel::execution::executor
    // threads::executor
    template <typename Executor>
    struct sync_dispatch<Executor,
        typename std::enable_if<traits::is_one_way_executor<Executor>::value ||
            traits::is_two_way_executor<Executor>::value
#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
            || traits::is_threads_executor<Executor>::value
#endif
            >::type>
    {
        template <typename Executor_, typename F, typename... Ts>
        HPX_FORCEINLINE static typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            typename util::detail::invoke_deferred_result<F, Ts...>::type>::type
        call(Executor_&& exec, F&& f, Ts&&... ts)
        {
            return parallel::execution::sync_execute(
                std::forward<Executor_>(exec), std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }
    };
}}    // namespace hpx::detail

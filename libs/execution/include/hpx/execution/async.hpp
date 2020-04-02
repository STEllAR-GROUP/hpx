//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_EXECUTION_ASYNC_APR_16_20012_0225PM)
#define HPX_EXECUTION_ASYNC_APR_16_20012_0225PM

#include <hpx/config.hpp>
#include <hpx/async_base/async.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/executors/parallel_executor.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/traits/is_launch_policy.hpp>

#include <utility>

namespace hpx { namespace detail {
    // TODO: Where are the launch policy overloads? In the async module...

    // Launch the given function or function object asynchronously and return a
    // future allowing to synchronize with the returned result.
    template <typename Func, typename Enable>
    struct async_dispatch
    {
        template <typename F, typename... Ts>
        HPX_FORCEINLINE static typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            hpx::future<typename util::detail::invoke_deferred_result<F,
                Ts...>::type>>::type
        call(F&& f, Ts&&... ts)
        {
            parallel::execution::parallel_executor exec;
            return exec.async_execute(
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }
    };

    // The overload for hpx::async taking an executor simply forwards to the
    // corresponding executor customization point.
    //
    // parallel::execution::executor
    // threads::executor
    template <typename Executor>
    struct async_dispatch<Executor,
        typename std::enable_if<traits::is_one_way_executor<Executor>::value ||
            traits::is_two_way_executor<Executor>::value ||
            traits::is_threads_executor<Executor>::value>::type>
    {
        template <typename Executor_, typename F, typename... Ts>
        HPX_FORCEINLINE static typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            hpx::future<typename util::detail::invoke_deferred_result<F,
                Ts...>::type>>::type
        call(Executor_&& exec, F&& f, Ts&&... ts)
        {
            return parallel::execution::async_execute(
                std::forward<Executor_>(exec), std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }
    };
}}    // namespace hpx::detail

#endif

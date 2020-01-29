//  Copyright (c) 2007-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SYNC_JUL_21_2018_0937PM)
#define HPX_SYNC_JUL_21_2018_0937PM

#include <hpx/config.hpp>
#include <hpx/sync_launch_policy_dispatch.hpp>
#include <hpx/lcos/sync.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/functional/traits/is_action.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/traits/is_launch_policy.hpp>
#include <hpx/util/bind_action.hpp>
#include <hpx/functional/deferred_call.hpp>

#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/executors/parallel_executor.hpp>

#include <exception>
#include <functional>
#include <type_traits>
#include <utility>

namespace hpx { namespace detail
{
    // Launch the given function or function object synchronously. This exists
    // mostly for symmetry with hpx::async.
    template <typename Func, typename Enable>
    struct sync_dispatch
    {
        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            typename util::detail::invoke_deferred_result<F, Ts...>::type
        >::type
        call(F && f, Ts &&... ts)
        {
            parallel::execution::parallel_executor exec;
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
        typename std::enable_if<
            traits::is_one_way_executor<Executor>::value ||
            traits::is_two_way_executor<Executor>::value ||
            traits::is_threads_executor<Executor>::value
        >::type>
    {
        template <typename Executor_, typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            typename util::detail::invoke_deferred_result<F, Ts...>::type
        >::type
        call(Executor_ && exec, F && f, Ts &&... ts)
        {
            return parallel::execution::sync_execute(
                std::forward<Executor_>(exec), std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }
    };

    // bound action
    template <typename Bound>
    struct sync_dispatch<Bound,
        typename std::enable_if<
            traits::is_bound_action<Bound>::value
        >::type>
    {
        template <typename Action, typename Is, typename... Ts, typename ...Us>
        HPX_FORCEINLINE
        static typename hpx::util::detail::bound_action<
            Action, Is, Ts...
        >::result_type
        call(hpx::util::detail::bound_action<Action, Is, Ts...> const& bound,
            Us&&... vs)
        {
            return bound(std::forward<Us>(vs)...);
        }
    };
}}

namespace hpx
{
    template <typename F, typename ...Ts>
    HPX_FORCEINLINE auto sync(F&& f, Ts&&... ts)
    ->  decltype(detail::sync_dispatch<typename util::decay<F>::type>::call(
            std::forward<F>(f), std::forward<Ts>(ts)...
        ))
    {
        return detail::sync_dispatch<
                typename util::decay<F>::type
            >::call(std::forward<F>(f), std::forward<Ts>(ts)...);
    }
}

#endif

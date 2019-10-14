//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_ASYNC_LAUNCH_POLICY_DISPATCH_NOV_26_2017_1243PM)
#define HPX_ASYNC_LAUNCH_POLICY_DISPATCH_NOV_26_2017_1243PM

#include <hpx/config.hpp>
#include <hpx/lcos/async_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/futures_factory.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/functional/traits/is_action.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/invoke.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace hpx { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename ...Ts>
    HPX_FORCEINLINE
    lcos::future<typename util::detail::invoke_deferred_result<F, Ts...>::type>
    call_sync(std::false_type, F f, Ts... vs) // decay-copy
    {
        using R = typename util::detail::invoke_deferred_result<F, Ts...>::type;
        try
        {
            return lcos::make_ready_future<R>(
                util::invoke(std::move(f), std::move(vs)...));
        }
        catch (...)
        {
            return lcos::make_exceptional_future<R>(
                std::current_exception());
        }
    }

    template <typename F, typename ...Ts>
    HPX_FORCEINLINE
    lcos::future<typename util::detail::invoke_deferred_result<F, Ts...>::type>
    call_sync(std::true_type, F f, Ts... vs) // decay-copy
    {
        try
        {
            util::invoke(std::move(f), std::move(vs)...);
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
        typename std::enable_if<
            !traits::is_action<Action>::value
        >::type>
    {
        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            hpx::future<
                typename util::detail::invoke_deferred_result<F, Ts...>::type
            >
        >::type
        call(launch policy, F && f, Ts&&... ts)
        {
            typedef typename util::detail::invoke_deferred_result<F, Ts...>::type
                result_type;

            if (policy == launch::sync)
            {
                using is_void = typename std::is_void<result_type>::type;
                return detail::call_sync(is_void{},
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            lcos::local::futures_factory<result_type()> p(
                util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...));
            if (hpx::detail::has_async_policy(policy))
            {
                threads::thread_id_type tid = p.apply(policy, policy.priority());
                if (tid && policy == launch::fork)
                {
                    // make sure this thread is executed last
                    // yield_to
                    hpx::this_thread::suspend(threads::pending, tid,
                        "async_launch_policy_dispatch<fork>");
                }
            }
            return p.get_future();
        }

        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            hpx::future<
                typename util::detail::invoke_deferred_result<F, Ts...>::type
            >
        >::type
        call(hpx::detail::sync_policy, F && f, Ts&&... ts)
        {
            typedef typename util::detail::invoke_deferred_result<F, Ts...>::type
                result_type;

            using is_void = typename std::is_void<result_type>::type;
            return detail::call_sync(is_void{},
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            hpx::future<
                typename util::detail::invoke_deferred_result<F, Ts...>::type
            >
        >::type
        call(hpx::detail::async_policy policy, F && f, Ts&&... ts)
        {
            typedef typename util::detail::invoke_deferred_result<F, Ts...>::type
                result_type;

            lcos::local::futures_factory<result_type()> p(
                util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...));

            p.apply(policy, policy.priority());
            return p.get_future();
        }

        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            hpx::future<
                typename util::detail::invoke_deferred_result<F, Ts...>::type
            >
        >::type
        call(hpx::detail::fork_policy policy, F && f, Ts&&... ts)
        {
            typedef typename util::detail::invoke_deferred_result<F, Ts...>::type
                result_type;

            lcos::local::futures_factory<result_type()> p(
                util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...));

            // make sure this thread is executed last
            threads::thread_id_type tid = p.apply(policy, policy.priority());
            if (tid)
            {
                // yield_to
                hpx::this_thread::suspend(threads::pending, tid,
                    "async_launch_policy_dispatch<fork>");
            }
            return p.get_future();
        }

        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value,
            hpx::future<
                typename util::detail::invoke_deferred_result<F, Ts...>::type
            >
        >::type
        call(hpx::detail::deferred_policy, F && f, Ts &&... ts)
        {
            typedef typename util::detail::invoke_deferred_result<F, Ts...>::type
                result_type;

            lcos::local::futures_factory<result_type()> p(
                util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...));

            return p.get_future();
        }
    };
}}

#endif

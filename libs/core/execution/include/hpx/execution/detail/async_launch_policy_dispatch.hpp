//  Copyright (c) 2007-2021 Hartmut Kaiser
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
#include <hpx/threading_base/thread_description.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace hpx { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    // dispatch point used for launch_policy implementations
    template <typename Action, typename Enable = void>
    struct async_launch_policy_dispatch;

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    HPX_FORCEINLINE hpx::future<
        util::detail::invoke_deferred_result_t<std::decay_t<F>, Ts...>>
    call_sync(F&& f, Ts... vs) noexcept
    {
        using R =
            util::detail::invoke_deferred_result_t<std::decay_t<F>, Ts...>;

        try
        {
            if constexpr (std::is_void_v<R>)
            {
                HPX_INVOKE(std::forward<F>(f), std::move(vs)...);
                return make_ready_future();
            }
            else
            {
                return make_ready_future<R>(
                    HPX_INVOKE(std::forward<F>(f), std::move(vs)...));
            }
        }
        catch (...)
        {
            return make_exceptional_future<R>(std::current_exception());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct async_launch_policy_dispatch<Action,
        std::enable_if_t<!traits::is_action_v<Action>>>
    {
        template <typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(launch policy, hpx::util::thread_description const& desc,
            threads::thread_pool_base* pool, F&& f, Ts&&... ts)
        {
            using result_type =
                util::detail::invoke_deferred_result_t<F, Ts...>;

            if (policy == launch::sync)
            {
                return detail::call_sync(
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            lcos::local::futures_factory<result_type()> p(util::deferred_call(
                std::forward<F>(f), std::forward<Ts>(ts)...));
            if (hpx::detail::has_async_policy(policy))
            {
                threads::thread_id_ref_type tid =
                    p.apply(pool, desc.get_description(), policy);
                if (tid)
                {
                    if (policy == launch::fork)
                    {
                        // make sure this thread is executed last
                        // yield_to
                        hpx::this_thread::suspend(
                            threads::thread_schedule_state::pending,
                            tid.noref(), desc.get_description());
                    }

                    auto&& result = p.get_future();
                    traits::detail::get_shared_state(result)->set_on_completed(
                        [tid = std::move(tid)]() { (void) tid; });
                    return std::move(result);
                }
            }
            return p.get_future();
        }

        template <typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(launch policy, hpx::util::thread_description const& desc, F&& f,
            Ts&&... ts)
        {
            return call(policy, desc,
                threads::detail::get_self_or_default_pool(), std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }

        template <typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(launch policy, F&& f, Ts&&... ts)
        {
            hpx::util::thread_description desc(f);
            return call(policy, desc,
                threads::detail::get_self_or_default_pool(), std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }

        template <typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(hpx::detail::sync_policy, F&& f, Ts&&... ts)
        {
            return detail::call_sync(
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(hpx::detail::async_policy policy,
            hpx::util::thread_description const& desc,
            threads::thread_pool_base* pool, F&& f, Ts&&... ts)
        {
            HPX_ASSERT(pool);

            using result_type =
                util::detail::invoke_deferred_result_t<F, Ts...>;

            lcos::local::futures_factory<result_type()> p(util::deferred_call(
                std::forward<F>(f), std::forward<Ts>(ts)...));

            threads::thread_id_ref_type tid =
                p.apply(pool, desc.get_description(), policy);

            if (tid)
            {
                // keep thread alive, if needed
                auto&& result = p.get_future();
                traits::detail::get_shared_state(result)->set_on_completed(
                    [tid = std::move(tid)]() { (void) tid; });
                return std::move(result);
            }
            return p.get_future();
        }

        template <typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(hpx::detail::async_policy policy, F&& f, Ts&&... ts)
        {
            hpx::util::thread_description desc(f);
            return call(policy, desc,
                threads::detail::get_self_or_default_pool(), std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }

        template <typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(hpx::detail::fork_policy policy,
            hpx::util::thread_description const& desc,
            threads::thread_pool_base* pool, F&& f, Ts&&... ts)
        {
            HPX_ASSERT(pool);

            using result_type =
                util::detail::invoke_deferred_result_t<F, Ts...>;

            lcos::local::futures_factory<result_type()> p(util::deferred_call(
                std::forward<F>(f), std::forward<Ts>(ts)...));

            threads::thread_id_ref_type tid =
                p.apply(pool, desc.get_description(), policy);

            // make sure this thread is executed last
            threads::thread_id_type tid_self = threads::get_self_id();
            if (tid && tid_self &&
                get_thread_id_data(tid)->get_scheduler_base() ==
                    get_thread_id_data(tid_self)->get_scheduler_base())
            {
                // yield_to
                hpx::this_thread::suspend(
                    threads::thread_schedule_state::pending, tid.noref(),
                    desc.get_description());

                // keep thread alive, if needed
                auto&& result = p.get_future();
                traits::detail::get_shared_state(result)->set_on_completed(
                    [tid = std::move(tid)]() { (void) tid; });
                return std::move(result);
            }
            return p.get_future();
        }

        template <typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(hpx::detail::fork_policy policy, F&& f, Ts&&... ts)
        {
            hpx::util::thread_description desc(f);
            return call(policy, desc,
                threads::detail::get_self_or_default_pool(), std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }

        template <typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(hpx::detail::deferred_policy, F&& f, Ts&&... ts)
        {
            using result_type =
                util::detail::invoke_deferred_result_t<F, Ts...>;

            lcos::local::futures_factory<result_type()> p(util::deferred_call(
                std::forward<F>(f), std::forward<Ts>(ts)...));

            return p.get_future();
        }
    };
}}    // namespace hpx::detail

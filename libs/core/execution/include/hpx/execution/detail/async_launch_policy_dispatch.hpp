//  Copyright (c) 2007-2022 Hartmut Kaiser
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
#include <hpx/threading_base/annotated_function.hpp>
#include <hpx/threading_base/scoped_annotation.hpp>
#include <hpx/threading_base/thread_description.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace hpx::detail {

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
                HPX_INVOKE(HPX_FORWARD(F, f), HPX_MOVE(vs)...);
                return make_ready_future();
            }
            else
            {
                return make_ready_future<R>(
                    HPX_INVOKE(HPX_FORWARD(F, f), HPX_MOVE(vs)...));
            }
        }
        catch (...)
        {
            return make_exceptional_future<R>(std::current_exception());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <>
    struct async_launch_policy_dispatch<hpx::launch::sync_policy>
    {
        template <typename Policy, typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(Policy const&, F&& f, Ts&&... ts)
        {
            return call_sync(HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        template <typename Policy, typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(Policy const&, hpx::threads::thread_description const& desc, F&& f,
            Ts&&... ts)
        {
            auto ann = hpx::scoped_annotation(desc.get_description());
            return call_sync(HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        template <typename Policy, typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(Policy const&, hpx::threads::thread_description const& desc,
            threads::thread_pool_base*, F&& f, Ts&&... ts)
        {
            auto ann = hpx::scoped_annotation(desc.get_description());
            return call_sync(HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }
    };

    template <>
    struct async_launch_policy_dispatch<hpx::launch::deferred_policy>
    {
        template <typename Policy, typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(Policy const&, F&& f, Ts&&... ts)
        {
            using result_type =
                util::detail::invoke_deferred_result_t<F, Ts...>;

            lcos::local::futures_factory<result_type()> p(
                util::deferred_call(HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...));
            return p.get_future();
        }

        template <typename Policy, typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(Policy const& policy, hpx::threads::thread_description const& desc,
            F&& f, Ts&&... ts)
        {
            return call(policy,
                hpx::annotated_function(
                    HPX_FORWARD(F, f), desc.get_description()),
                HPX_FORWARD(Ts, ts)...);
        }

        template <typename Policy, typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(Policy const& policy, hpx::threads::thread_description const& desc,
            threads::thread_pool_base*, F&& f, Ts&&... ts)
        {
            return call(policy,
                hpx::annotated_function(
                    HPX_FORWARD(F, f), desc.get_description()),
                HPX_FORWARD(Ts, ts)...);
        }
    };

    template <>
    struct async_launch_policy_dispatch<hpx::launch::async_policy>
    {
        template <typename Policy, typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(Policy const& policy, hpx::threads::thread_description const& desc,
            threads::thread_pool_base* pool, F&& f, Ts&&... ts)
        {
            HPX_ASSERT(pool);

            using result_type =
                util::detail::invoke_deferred_result_t<F, Ts...>;

            lcos::local::futures_factory<result_type()> p(
                util::deferred_call(HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...));

            threads::thread_id_ref_type tid =
                p.post(pool, desc.get_description(), policy);

            if (tid)
            {
                // keep thread alive, if needed
                auto&& result = p.get_future();
                traits::detail::get_shared_state(result)->set_on_completed(
                    [tid = HPX_MOVE(tid)]() { (void) tid; });
                return HPX_MOVE(result);
            }

            return p.get_future();
        }

        template <typename Policy, typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(Policy const& policy, hpx::threads::thread_description const& desc,
            F&& f, Ts&&... ts)
        {
            return call(policy, desc,
                threads::detail::get_self_or_default_pool(), HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
        }

        template <typename Policy, typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(Policy const& policy, F&& f, Ts&&... ts)
        {
            hpx::threads::thread_description desc(f);
            return call(policy, desc,
                threads::detail::get_self_or_default_pool(), HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
        }
    };

    template <>
    struct async_launch_policy_dispatch<hpx::launch::fork_policy>
    {
        template <typename Policy, typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(Policy const& policy, hpx::threads::thread_description const& desc,
            threads::thread_pool_base* pool, F&& f, Ts&&... ts)
        {
            HPX_ASSERT(pool);

            using result_type =
                util::detail::invoke_deferred_result_t<F, Ts...>;

            lcos::local::futures_factory<result_type()> p(
                util::deferred_call(HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...));

            threads::thread_id_ref_type tid =
                p.post(pool, desc.get_description(), policy);

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
                    [tid = HPX_MOVE(tid)]() { (void) tid; });
                return HPX_MOVE(result);
            }

            return p.get_future();
        }

        template <typename Policy, typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(Policy const& policy, hpx::threads::thread_description const& desc,
            F&& f, Ts&&... ts)
        {
            return call(policy, desc,
                threads::detail::get_self_or_default_pool(), HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
        }

        template <typename Policy, typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(Policy const& policy, F&& f, Ts&&... ts)
        {
            hpx::threads::thread_description desc(f);
            return call(policy, desc,
                threads::detail::get_self_or_default_pool(), HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
        }
    };

    template <typename Action>
    struct async_launch_policy_dispatch<Action,
        std::enable_if_t<!traits::is_action_v<Action>>>
    {
        template <typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(launch policy, hpx::threads::thread_description const& desc,
            threads::thread_pool_base* pool, F&& f, Ts&&... ts)
        {
            if (policy == launch::sync)
            {
                return async_launch_policy_dispatch<
                    hpx::launch::sync_policy>::call(policy, desc, pool,
                    HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
            }
            if (policy == launch::deferred)
            {
                return async_launch_policy_dispatch<
                    hpx::launch::deferred_policy>::call(policy, desc, pool,
                    HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
            }
            if (policy == launch::fork)
            {
                return async_launch_policy_dispatch<
                    hpx::launch::fork_policy>::call(policy, desc, pool,
                    HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
            }

            return async_launch_policy_dispatch<
                hpx::launch::async_policy>::call(policy, desc, pool,
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        template <typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(launch policy, hpx::threads::thread_description const& desc, F&& f,
            Ts&&... ts)
        {
            return call(policy, desc,
                threads::detail::get_self_or_default_pool(), HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
        }

        template <typename F, typename... Ts>
        HPX_FORCEINLINE static std::enable_if_t<
            traits::detail::is_deferred_invocable_v<F, Ts...>,
            hpx::future<util::detail::invoke_deferred_result_t<F, Ts...>>>
        call(launch policy, F&& f, Ts&&... ts)
        {
            hpx::threads::thread_description desc(f);
            return call(policy, desc,
                threads::detail::get_self_or_default_pool(), HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
        }
    };
}    // namespace hpx::detail

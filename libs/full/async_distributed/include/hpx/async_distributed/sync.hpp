//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/actions_base/traits/is_client.hpp>
#include <hpx/actions_base/traits/is_valid_action.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_base/traits/is_launch_policy.hpp>
#include <hpx/async_distributed/bind_action.hpp>
#include <hpx/async_distributed/detail/sync_implementations.hpp>
#include <hpx/async_distributed/sync.hpp>
#include <hpx/async_local/sync.hpp>
#include <hpx/components/client_base.hpp>
#include <hpx/execution/detail/sync_launch_policy_dispatch.hpp>
#include <hpx/executors/sync.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/functional/traits/is_action.hpp>
#include <hpx/futures/traits/future_traits.hpp>
#include <hpx/futures/traits/is_future.hpp>
#include <hpx/futures/traits/promise_local_result.hpp>
#include <hpx/naming_base/id_type.hpp>

#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::detail {

    template <typename Action>
    struct sync_result
    {
        using type = traits::promise_local_result_t<
            typename traits::extract_action<Action>::remote_result_type>;
    };

    template <typename Action>
    using sync_result_t = typename sync_result<Action>::type;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct sync_action_client_dispatch
    {
        template <typename Policy, typename Client, typename Stub,
            typename Data, typename... Ts>
        HPX_FORCEINLINE std::enable_if_t<traits::is_launch_policy_v<Policy>,
            sync_result_t<Action>>
        operator()(components::client_base<Client, Stub, Data> const& c,
            Policy const& launch_policy, Ts&&... ts) const
        {
            HPX_ASSERT(c.is_ready());
            return hpx::detail::sync_impl<Action>(
                launch_policy, c.get_id(), HPX_FORWARD(Ts, ts)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // launch
    template <typename Action, typename Policy>
    struct sync_action_dispatch<Action, Policy,
        std::enable_if_t<traits::is_launch_policy_v<Policy>>>
    {
        // id_type
        template <typename Policy_, typename... Ts>
        HPX_FORCEINLINE static sync_result_t<Action> call(
            Policy_&& launch_policy, hpx::id_type const& id, Ts&&... ts)
        {
            return hpx::detail::sync_impl<Action>(
                HPX_FORWARD(Policy_, launch_policy), id,
                HPX_FORWARD(Ts, ts)...);
        }

        template <typename Policy_, typename Client, typename Stub,
            typename Data, typename... Ts>
        HPX_FORCEINLINE static sync_result_t<Action> call(
            Policy_&& launch_policy,
            components::client_base<Client, Stub, Data> const& c, Ts&&... ts)
        {
            // make sure the action is compatible with the component type
            using component_type = typename components::client_base<Client,
                Stub, Data>::server_component_type;

            static_assert(traits::is_valid_action_v<Action, component_type>,
                "The action to invoke is not supported by the target");

            // invoke directly if client is ready
            if (c.is_ready())
            {
                return hpx::detail::sync_impl<Action>(
                    HPX_FORWARD(Policy_, launch_policy), c.get_id(),
                    HPX_FORWARD(Ts, ts)...);
            }

            // defer invocation otherwise
            return c.then(launch_policy,
                util::one_shot(
                    hpx::bind_back(sync_action_client_dispatch<Action>(),
                        launch_policy, HPX_FORWARD(Ts, ts)...)));
        }
    };

    // hpx::id_type
    template <typename Action>
    struct sync_action_dispatch<Action, hpx::id_type>
    {
        template <typename... Ts>
        HPX_FORCEINLINE static decltype(auto) call(
            hpx::id_type const& id, Ts&&... ts)
        {
            return sync_action_dispatch<Action, hpx::detail::sync_policy>::call(
                launch::sync, id, HPX_FORWARD(Ts, ts)...);
        }
    };

    // component::client
    template <typename Action, typename Client>
    struct sync_action_dispatch<Action, Client,
        std::enable_if_t<traits::is_client_v<Client>>>
    {
        template <typename Client_, typename Stub, typename Data,
            typename... Ts>
        HPX_FORCEINLINE static decltype(auto) call(
            components::client_base<Client_, Stub, Data> const& c, Ts&&... ts)
        {
            return sync_action_dispatch<Action, hpx::detail::sync_policy>::call(
                launch::sync, c, HPX_FORWARD(Ts, ts)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct sync_launch_policy_dispatch<Action,
        std::enable_if_t<traits::is_action_v<Action>>>
    {
        template <typename Policy, typename... Ts>
        HPX_FORCEINLINE static decltype(auto) call(
            Policy&& launch_policy, Action const&, Ts&&... ts)
        {
            static_assert(traits::is_launch_policy_v<std::decay_t<Policy>>,
                "Policy must be a valid launch policy");

            return sync<Action>(
                HPX_FORWARD(Policy, launch_policy), HPX_FORWARD(Ts, ts)...);
        }
    };
}    // namespace hpx::detail

namespace hpx {

    template <typename Action, typename F, typename... Ts>
    HPX_FORCEINLINE auto sync(F&& f, Ts&&... ts)
        -> decltype(detail::sync_action_dispatch<Action, std::decay_t<F>>::call(
            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...))
    {
        return detail::sync_action_dispatch<Action, std::decay_t<F>>::call(
            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx

///////////////////////////////////////////////////////////////////////////////
namespace hpx::detail {

    ///////////////////////////////////////////////////////////////////////////
    // any action
    template <typename Action>
    struct sync_dispatch<Action, std::enable_if_t<traits::is_action_v<Action>>>
    {
        template <typename Component, typename Signature, typename Derived,
            typename... Ts>
        HPX_FORCEINLINE static decltype(auto) call(
            hpx::actions::basic_action<Component, Signature, Derived> const&,
            hpx::id_type const& id, Ts&&... vs)
        {
            return sync<Derived>(launch::sync, id, HPX_FORWARD(Ts, vs)...);
        }

        template <typename Component, typename Signature, typename Derived,
            typename Client, typename Stub, typename Data, typename... Ts>
        HPX_FORCEINLINE static decltype(auto) call(
            hpx::actions::basic_action<Component, Signature, Derived> const&,
            components::client_base<Client, Stub, Data> const& c, Ts&&... vs)
        {
            using component_type = typename components::client_base<Client,
                Stub, Data>::server_component_type;

            static_assert(traits::is_valid_action_v<Action, component_type>,
                "The action to invoke is not supported by the target");

            return sync<Derived>(
                launch::sync, c.get_id(), HPX_FORWARD(Ts, vs)...);
        }
    };

    // launch with any action
    template <typename Func>
    struct sync_dispatch_launch_policy_helper<Func,
        std::enable_if_t<traits::is_action_v<Func>>>
    {
        template <typename Policy_, typename F, typename... Ts>
        HPX_FORCEINLINE static auto call(
            Policy_&& launch_policy, F&& f, Ts&&... ts)
            -> decltype(sync_launch_policy_dispatch<std::decay_t<F>>::call(
                HPX_FORWARD(Policy_, launch_policy), HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...))
        {
            return sync_launch_policy_dispatch<std::decay_t<F>>::call(
                HPX_FORWARD(Policy_, launch_policy), HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
        }

        template <typename Policy_, typename Component, typename Signature,
            typename Derived, typename Client, typename Stub, typename Data,
            typename... Ts>
        HPX_FORCEINLINE static sync_result_t<Derived> call(
            Policy_&& launch_policy,
            hpx::actions::basic_action<Component, Signature, Derived> const&,
            components::client_base<Client, Stub, Data> const& c, Ts&&... ts)
        {
            using component_type = typename components::client_base<Client,
                Stub, Data>::server_component_type;

            static_assert(traits::is_valid_action_v<Derived, component_type>,
                "The action to invoke is not supported by the target");

            return sync<Derived>(HPX_FORWARD(Policy_, launch_policy),
                c.get_id(), HPX_FORWARD(Ts, ts)...);
        }
    };

    // bound action
    template <typename Bound>
    struct sync_dispatch<Bound, std::enable_if_t<hpx::is_bound_action_v<Bound>>>
    {
        template <typename Action, typename Is, typename... Ts, typename... Us>
        HPX_FORCEINLINE static decltype(auto) call(
            hpx::detail::bound_action<Action, Is, Ts...> const& bound,
            Us&&... vs)
        {
            return bound(HPX_FORWARD(Us, vs)...);
        }
    };
}    // namespace hpx::detail

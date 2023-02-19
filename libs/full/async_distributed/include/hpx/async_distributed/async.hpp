//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/actions_base/traits/is_client.hpp>
#include <hpx/actions_base/traits/is_distribution_policy.hpp>
#include <hpx/actions_base/traits/is_valid_action.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_base/traits/is_launch_policy.hpp>
#include <hpx/async_distributed/async.hpp>
#include <hpx/async_distributed/async_continue.hpp>
#include <hpx/async_distributed/bind_action.hpp>
#include <hpx/async_distributed/detail/async_implementations.hpp>
#include <hpx/async_local/async.hpp>
#include <hpx/components/client_base.hpp>
#include <hpx/execution/detail/async_launch_policy_dispatch.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/functional/traits/is_action.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/promise_local_result.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/type_support/identity.hpp>
#include <hpx/type_support/lazy_enable_if.hpp>

#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct async_action_client_dispatch
    {
        template <typename Policy, typename Client, typename Stub,
            typename Data, typename... Ts>
        HPX_FORCEINLINE std::enable_if_t<traits::is_launch_policy_v<Policy>,
            hpx::future<typename traits::promise_local_result<typename traits::
                    extract_action<Action>::remote_result_type>::type>>
        operator()(components::client_base<Client, Stub, Data> const& c,
            Policy const& launch_policy, Ts&&... ts) const
        {
            HPX_ASSERT(c.is_ready());
            return hpx::detail::async_impl<Action>(
                launch_policy, c.get_id(), HPX_FORWARD(Ts, ts)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // launch
    template <typename Action, typename Policy>
    struct async_action_dispatch<Action, Policy,
        std::enable_if_t<traits::is_launch_policy_v<Policy>>>
    {
        using result_type =
            hpx::future<typename traits::promise_local_result<typename hpx::
                    traits::extract_action<Action>::remote_result_type>::type>;

        // id_type
        template <typename Policy_, typename... Ts>
        HPX_FORCEINLINE static result_type call(
            Policy_&& launch_policy, hpx::id_type const& id, Ts&&... ts)
        {
            return hpx::detail::async_impl<Action>(
                HPX_FORWARD(Policy_, launch_policy), id,
                HPX_FORWARD(Ts, ts)...);
        }

        template <typename Policy_, typename Client, typename Stub,
            typename Data, typename... Ts>
        HPX_FORCEINLINE static result_type call(Policy_&& launch_policy,
            components::client_base<Client, Stub, Data> c, Ts&&... ts)
        {
            // make sure the action is compatible with the component type
            using component_type = typename components::client_base<Client,
                Stub, Data>::server_component_type;

            static_assert(traits::is_valid_action_v<Action, component_type>,
                "The action to invoke is not supported by the target");

            // invoke directly if client is ready
            if (c.is_ready())
            {
                return hpx::detail::async_impl<Action>(
                    HPX_FORWARD(Policy_, launch_policy), c.get_id(),
                    HPX_FORWARD(Ts, ts)...);
            }

            // defer invocation otherwise
            return c.then(util::one_shot(hpx::bind_back(
                async_action_client_dispatch<Action>(),
                HPX_FORWARD(Policy_, launch_policy), HPX_FORWARD(Ts, ts)...)));
        }

        // distribution policy
        template <typename Policy_, typename DistPolicy, typename... Ts>
        HPX_FORCEINLINE static typename util::lazy_enable_if<
            traits::is_distribution_policy_v<DistPolicy>,
            typename DistPolicy::template async_result<Action>>::type
        call(Policy_&& launch_policy, DistPolicy const& policy, Ts&&... ts)
        {
            return policy.template async<Action>(
                HPX_FORWARD(Policy_, launch_policy), HPX_FORWARD(Ts, ts)...);
        }
    };

    // hpx::id_type
    template <typename Action>
    struct async_action_dispatch<Action, hpx::id_type>
    {
        template <typename... Ts>
        HPX_FORCEINLINE static hpx::future<
            typename traits::promise_local_result<typename hpx::traits::
                    extract_action<Action>::remote_result_type>::type>
        call(hpx::id_type const& id, Ts&&... ts)
        {
            return async_action_dispatch<Action,
                hpx::detail::async_policy>::call(launch::async, id,
                HPX_FORWARD(Ts, ts)...);
        }
    };

    // component::client
    template <typename Action, typename Client>
    struct async_action_dispatch<Action, Client,
        std::enable_if_t<traits::is_client<Client>::value>>
    {
        template <typename Client_, typename Stub, typename Data,
            typename... Ts>
        HPX_FORCEINLINE static hpx::future<
            typename traits::promise_local_result<typename traits::
                    extract_action<Action>::remote_result_type>::type>
        call(components::client_base<Client_, Stub, Data> const& c, Ts&&... ts)
        {
            return async_action_dispatch<Action,
                hpx::detail::async_policy>::call(launch::async, c,
                HPX_FORWARD(Ts, ts)...);
        }
    };

    // distribution policy
    template <typename Action, typename Policy>
    struct async_action_dispatch<Action, Policy,
        std::enable_if_t<traits::is_distribution_policy_v<Policy>>>
    {
        template <typename DistPolicy, typename... Ts>
        HPX_FORCEINLINE static decltype(auto) call(
            DistPolicy const& policy, Ts&&... ts)
        {
            return async_action_dispatch<Action,
                hpx::detail::async_policy>::call(launch::async, policy,
                HPX_FORWARD(Ts, ts)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct async_launch_policy_dispatch<Action,
        std::enable_if_t<traits::is_action<Action>::value>>
    {
        using result_type = typename traits::promise_local_result<typename hpx::
                traits::extract_action<Action>::remote_result_type>::type;

        template <typename Policy, typename... Ts>
        HPX_FORCEINLINE static auto call(
            Policy&& launch_policy, Action const&, Ts&&... ts)
            -> decltype(async<Action>(
                HPX_FORWARD(Policy, launch_policy), HPX_FORWARD(Ts, ts)...))
        {
            static_assert(traits::is_launch_policy_v<std::decay_t<Policy>>,
                "Policy must be a valid launch policy");
            return async<Action>(
                HPX_FORWARD(Policy, launch_policy), HPX_FORWARD(Ts, ts)...);
        }
    };
}    // namespace hpx::detail

namespace hpx {

    // different versions of clang-format disagree
    // clang-format off
    template <typename Action, typename F, typename... Ts>
    HPX_FORCEINLINE auto async(F&& f, Ts&&... ts) -> decltype(
        detail::async_action_dispatch<Action, std::decay_t<F>>::call(
            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...))
    // clang-format on
    {
        return detail::async_action_dispatch<Action, std::decay_t<F>>::call(
            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx

///////////////////////////////////////////////////////////////////////////////
namespace hpx::detail {

    ///////////////////////////////////////////////////////////////////////////
    // any action
    template <typename Action>
    struct async_dispatch<Action,
        std::enable_if_t<traits::is_action<Action>::value>>
    {
        template <typename Component, typename Signature, typename Derived,
            typename... Ts>
        HPX_FORCEINLINE static hpx::future<
            typename traits::promise_local_result<typename hpx::traits::
                    extract_action<Derived>::remote_result_type>::type>
        call(hpx::actions::basic_action<Component, Signature, Derived> const&,
            hpx::id_type const& id, Ts&&... vs)
        {
            return async<Derived>(launch::async, id, HPX_FORWARD(Ts, vs)...);
        }

        template <typename Component, typename Signature, typename Derived,
            typename Client, typename Stub, typename Data, typename... Ts>
        HPX_FORCEINLINE static hpx::future<
            typename traits::promise_local_result<typename traits::
                    extract_action<Derived>::remote_result_type>::type>
        call(hpx::actions::basic_action<Component, Signature, Derived> const&,
            components::client_base<Client, Stub, Data> const& c, Ts&&... vs)
        {
            using component_type = typename components::client_base<Client,
                Stub, Data>::server_component_type;

            static_assert(traits::is_valid_action_v<Derived, component_type>,
                "The action to invoke is not supported by the target");

            return async<Derived>(
                launch::async, c.get_id(), HPX_FORWARD(Ts, vs)...);
        }

        template <typename Component, typename Signature, typename Derived,
            typename DistPolicy, typename... Ts>
        HPX_FORCEINLINE static decltype(auto) call(
            hpx::actions::basic_action<Component, Signature, Derived> const&,
            DistPolicy const& policy, Ts&&... vs)
        {
            return async<Derived>(policy, HPX_FORWARD(Ts, vs)...);
        }
    };

    // launch with any action
    template <typename Func>
    struct async_dispatch_launch_policy_helper<Func,
        std::enable_if_t<traits::is_action<Func>::value>>
    {
        // clang-format off
        template <typename Policy_, typename F, typename... Ts>
        HPX_FORCEINLINE static auto call(
            Policy_&& launch_policy, F&& f, Ts&&... ts)
            -> decltype(
                detail::async_launch_policy_dispatch<std::decay_t<F>>::call(
                    HPX_FORWARD(Policy_, launch_policy), HPX_FORWARD(F, f),
                    HPX_FORWARD(Ts, ts)...))
        // clang-format on
        {
            return detail::async_launch_policy_dispatch<std::decay_t<F>>::call(
                HPX_FORWARD(Policy_, launch_policy), HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
        }

        template <typename Policy_, typename Component, typename Signature,
            typename Derived, typename Client, typename Stub, typename Data,
            typename... Ts>
        HPX_FORCEINLINE static decltype(auto) call(Policy_&& launch_policy,
            hpx::actions::basic_action<Component, Signature, Derived> const&,
            components::client_base<Client, Stub, Data> const& c, Ts&&... ts)
        {
            using component_type = typename components::client_base<Client,
                Stub, Data>::server_component_type;

            static_assert(traits::is_valid_action_v<Derived, component_type>,
                "The action to invoke is not supported by the target");

            return async<Derived>(HPX_FORWARD(Policy_, launch_policy),
                c.get_id(), HPX_FORWARD(Ts, ts)...);
        }

        template <typename Policy_, typename Component, typename Signature,
            typename Derived, typename DistPolicy, typename... Ts>
        HPX_FORCEINLINE static typename util::lazy_enable_if<
            traits::is_distribution_policy_v<DistPolicy>,
            typename DistPolicy::template async_result<Derived>>::type
        call(Policy_&& launch_policy,
            hpx::actions::basic_action<Component, Signature, Derived> const&,
            DistPolicy const& policy, Ts&&... ts)
        {
            return async<Derived>(HPX_FORWARD(Policy_, launch_policy), policy,
                HPX_FORWARD(Ts, ts)...);
        }
    };
}    // namespace hpx::detail

#include <hpx/async_distributed/sync.hpp>

namespace hpx::detail {

    // bound action
    template <typename Bound>
    struct async_dispatch<Bound,
        std::enable_if_t<hpx::is_bound_action_v<Bound>>>
    {
        template <typename Action, typename Is, typename... Ts, typename... Us>
        HPX_FORCEINLINE static hpx::future<
            typename hpx::detail::bound_action<Action, Is, Ts...>::result_type>
        call(hpx::detail::bound_action<Action, Is, Ts...> const& bound,
            Us&&... vs)
        {
            return bound.async(HPX_FORWARD(Us, vs)...);
        }
    };
}    // namespace hpx::detail

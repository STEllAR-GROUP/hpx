//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/actions_base/traits/is_valid_action.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_base/traits/is_launch_policy.hpp>
#include <hpx/async_distributed/async.hpp>
#include <hpx/async_distributed/async_continue.hpp>
#include <hpx/async_distributed/detail/async_implementations.hpp>
#include <hpx/async_local/async.hpp>
#include <hpx/execution/detail/async_launch_policy_dispatch.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/executors/parallel_executor.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/traits/is_action.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/promise_local_result.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/traits/is_client.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/type_support/lazy_enable_if.hpp>
#include <hpx/util/bind_action.hpp>

#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct async_action_client_dispatch
    {
        template <typename Policy, typename Client, typename Stub,
            typename... Ts>
        HPX_FORCEINLINE typename std::enable_if<
            traits::is_launch_policy<Policy>::value,
            lcos::future<typename traits::promise_local_result<typename traits::
                    extract_action<Action>::remote_result_type>::type>>::type
        operator()(components::client_base<Client, Stub> const& c,
            Policy const& launch_policy, Ts&&... ts) const
        {
            HPX_ASSERT(c.is_ready());
            return hpx::detail::async_impl<Action>(
                launch_policy, c.get_id(), std::forward<Ts>(ts)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // launch
    template <typename Action, typename Policy>
    struct async_action_dispatch<Action, Policy,
        typename std::enable_if<traits::is_launch_policy<Policy>::value>::type>
    {
        // id_type
        template <typename Policy_, typename... Ts>
        HPX_FORCEINLINE static lcos::future<
            typename traits::promise_local_result<typename hpx::traits::
                    extract_action<Action>::remote_result_type>::type>
        call(Policy_&& launch_policy, naming::id_type const& id, Ts&&... ts)
        {
            return hpx::detail::async_impl<Action>(
                std::forward<Policy_>(launch_policy), id,
                std::forward<Ts>(ts)...);
        }

        template <typename Policy_, typename Client, typename Stub,
            typename... Ts>
        HPX_FORCEINLINE static lcos::future<
            typename traits::promise_local_result<typename traits::
                    extract_action<Action>::remote_result_type>::type>
        call(Policy_&& launch_policy, components::client_base<Client, Stub> c,
            Ts&&... ts)
        {
            // make sure the action is compatible with the component type
            typedef typename components::client_base<Client,
                Stub>::server_component_type component_type;

            typedef traits::is_valid_action<Action, component_type> is_valid;
            static_assert(is_valid::value,
                "The action to invoke is not supported by the target");

            // invoke directly if client is ready
            if (c.is_ready())
            {
                return hpx::detail::async_impl<Action>(
                    std::forward<Policy_>(launch_policy), c.get_id(),
                    std::forward<Ts>(ts)...);
            }

            // defer invocation otherwise
            return c.then(util::one_shot(
                util::bind_back(async_action_client_dispatch<Action>(),
                    std::forward<Policy_>(launch_policy),
                    std::forward<Ts>(ts)...)));
        }

        // distribution policy
        template <typename Policy_, typename DistPolicy, typename... Ts>
        HPX_FORCEINLINE static typename util::lazy_enable_if<
            traits::is_distribution_policy<DistPolicy>::value,
            typename DistPolicy::template async_result<Action>>::type
        call(Policy_&& launch_policy, DistPolicy const& policy, Ts&&... ts)
        {
            return policy.template async<Action>(
                std::forward<Policy_>(launch_policy), std::forward<Ts>(ts)...);
        }
    };

    // naming::id_type
    template <typename Action>
    struct async_action_dispatch<Action, naming::id_type>
    {
        template <typename... Ts>
        HPX_FORCEINLINE static lcos::future<
            typename traits::promise_local_result<typename hpx::traits::
                    extract_action<Action>::remote_result_type>::type>
        call(naming::id_type const& id, Ts&&... ts)
        {
            return async_action_dispatch<Action,
                hpx::detail::async_policy>::call(launch::async, id,
                std::forward<Ts>(ts)...);
        }
    };

    // component::client
    template <typename Action, typename Client>
    struct async_action_dispatch<Action, Client,
        typename std::enable_if<traits::is_client<Client>::value>::type>
    {
        template <typename Client_, typename Stub, typename... Ts>
        HPX_FORCEINLINE static lcos::future<
            typename traits::promise_local_result<typename traits::
                    extract_action<Action>::remote_result_type>::type>
        call(components::client_base<Client_, Stub> const& c, Ts&&... ts)
        {
            return async_action_dispatch<Action,
                hpx::detail::async_policy>::call(launch::async, c,
                std::forward<Ts>(ts)...);
        }
    };

    // distribution policy
    template <typename Action, typename Policy>
    struct async_action_dispatch<Action, Policy,
        typename std::enable_if<
            traits::is_distribution_policy<Policy>::value>::type>
    {
        template <typename DistPolicy, typename... Ts>
        HPX_FORCEINLINE static
            typename DistPolicy::template async_result<Action>::type
            call(DistPolicy const& policy, Ts&&... ts)
        {
            return async_action_dispatch<Action,
                hpx::detail::async_policy>::call(launch::async, policy,
                std::forward<Ts>(ts)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct async_launch_policy_dispatch<Action,
        typename std::enable_if<traits::is_action<Action>::value>::type>
    {
        typedef typename traits::promise_local_result<typename hpx::traits::
                extract_action<Action>::remote_result_type>::type result_type;

        template <typename Policy, typename... Ts>
        HPX_FORCEINLINE static auto call(
            Policy&& launch_policy, Action const&, Ts&&... ts)
            -> decltype(async<Action>(
                std::forward<Policy>(launch_policy), std::forward<Ts>(ts)...))
        {
            static_assert(traits::is_launch_policy<
                              typename std::decay<Policy>::type>::value,
                "Policy must be a valid launch policy");
            return async<Action>(
                std::forward<Policy>(launch_policy), std::forward<Ts>(ts)...);
        }
    };
}}    // namespace hpx::detail

namespace hpx {
    template <typename Action, typename F, typename... Ts>
    HPX_FORCEINLINE auto async(F&& f, Ts&&... ts)
        -> decltype(detail::async_action_dispatch<Action,
            typename std::decay<F>::type>::call(std::forward<F>(f),
            std::forward<Ts>(ts)...))
    {
        return detail::async_action_dispatch<Action,
            typename std::decay<F>::type>::call(std::forward<F>(f),
            std::forward<Ts>(ts)...);
    }
}    // namespace hpx

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    // any action
    template <typename Action>
    struct async_dispatch<Action,
        typename std::enable_if<traits::is_action<Action>::value>::type>
    {
        template <typename Component, typename Signature, typename Derived,
            typename... Ts>
        HPX_FORCEINLINE static lcos::future<
            typename traits::promise_local_result<typename hpx::traits::
                    extract_action<Derived>::remote_result_type>::type>
        call(hpx::actions::basic_action<Component, Signature, Derived> const&,
            naming::id_type const& id, Ts&&... vs)
        {
            return async<Derived>(launch::async, id, std::forward<Ts>(vs)...);
        }

        template <typename Component, typename Signature, typename Derived,
            typename Client, typename Stub, typename... Ts>
        HPX_FORCEINLINE static lcos::future<
            typename traits::promise_local_result<typename traits::
                    extract_action<Derived>::remote_result_type>::type>
        call(hpx::actions::basic_action<Component, Signature, Derived> const&,
            components::client_base<Client, Stub> const& c, Ts&&... vs)
        {
            typedef typename components::client_base<Client,
                Stub>::server_component_type component_type;

            typedef traits::is_valid_action<Derived, component_type> is_valid;
            static_assert(is_valid::value,
                "The action to invoke is not supported by the target");

            return async<Derived>(
                launch::async, c.get_id(), std::forward<Ts>(vs)...);
        }

        template <typename Component, typename Signature, typename Derived,
            typename DistPolicy, typename... Ts>
        HPX_FORCEINLINE static
            typename DistPolicy::template async_result<Derived>::type
            call(hpx::actions::basic_action<Component, Signature,
                     Derived> const&,
                DistPolicy const& policy, Ts&&... vs)
        {
            return async<Derived>(policy, std::forward<Ts>(vs)...);
        }
    };

    // launch with any action
    template <typename Func>
    struct async_dispatch_launch_policy_helper<Func,
        typename std::enable_if<traits::is_action<Func>::value>::type>
    {
        template <typename Policy_, typename F, typename... Ts>
        HPX_FORCEINLINE static auto call(
            Policy_&& launch_policy, F&& f, Ts&&... ts)
            -> decltype(
                detail::async_launch_policy_dispatch<typename std::decay<
                    F>::type>::call(std::forward<Policy_>(launch_policy),
                    std::forward<F>(f), std::forward<Ts>(ts)...))
        {
            return detail::async_launch_policy_dispatch<typename std::decay<
                F>::type>::call(std::forward<Policy_>(launch_policy),
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename Policy_, typename Component, typename Signature,
            typename Derived, typename Client, typename Stub, typename... Ts>
        HPX_FORCEINLINE static lcos::future<
            typename traits::promise_local_result<typename traits::
                    extract_action<Derived>::remote_result_type>::type>
        call(Policy_&& launch_policy,
            hpx::actions::basic_action<Component, Signature, Derived> const&,
            components::client_base<Client, Stub> const& c, Ts&&... ts)
        {
            typedef typename components::client_base<Client,
                Stub>::server_component_type component_type;

            typedef traits::is_valid_action<Derived, component_type> is_valid;
            static_assert(is_valid::value,
                "The action to invoke is not supported by the target");

            return async<Derived>(std::forward<Policy_>(launch_policy),
                c.get_id(), std::forward<Ts>(ts)...);
        }

        template <typename Policy_, typename Component, typename Signature,
            typename Derived, typename DistPolicy, typename... Ts>
        HPX_FORCEINLINE static typename util::lazy_enable_if<
            traits::is_distribution_policy<DistPolicy>::value,
            typename DistPolicy::template async_result<Derived>>::type
        call(Policy_&& launch_policy,
            hpx::actions::basic_action<Component, Signature, Derived> const&,
            DistPolicy const& policy, Ts&&... ts)
        {
            return async<Derived>(std::forward<Policy_>(launch_policy), policy,
                std::forward<Ts>(ts)...);
        }
    };
}}    // namespace hpx::detail

#include <hpx/async_distributed/sync.hpp>

namespace hpx { namespace detail {
    // bound action
    template <typename Bound>
    struct async_dispatch<Bound,
        typename std::enable_if<traits::is_bound_action<Bound>::value>::type>
    {
        template <typename Action, typename Is, typename... Ts, typename... Us>
        HPX_FORCEINLINE static hpx::future<typename hpx::util::detail::
                bound_action<Action, Is, Ts...>::result_type>
        call(hpx::util::detail::bound_action<Action, Is, Ts...> const& bound,
            Us&&... vs)
        {
            return bound.async(std::forward<Us>(vs)...);
        }
    };
}}    // namespace hpx::detail

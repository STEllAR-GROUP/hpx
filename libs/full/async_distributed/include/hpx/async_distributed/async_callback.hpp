//  Copyright (c) 2007-2015 Hartmut Kaiser
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
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_base/traits/is_launch_policy.hpp>
#include <hpx/async_distributed/async_callback_fwd.hpp>
#include <hpx/async_distributed/detail/async_implementations_fwd.hpp>
#include <hpx/components/client_base.hpp>
#include <hpx/functional/traits/is_action.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/promise_local_result.hpp>

#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace detail {
    // launch
    template <typename Action, typename Policy>
    struct async_cb_action_dispatch<Action, Policy,
        typename std::enable_if<traits::is_launch_policy<Policy>::value>::type>
    {
        // id_type
        template <typename Policy_, typename Callback, typename... Ts>
        HPX_FORCEINLINE static hpx::future<
            typename traits::promise_local_result<typename traits::
                    extract_action<Action>::remote_result_type>::type>
        call(Policy_&& launch_policy, hpx::id_type const& id, Callback&& cb,
            Ts&&... ts)
        {
            return hpx::detail::async_cb_impl<Action>(
                HPX_FORWARD(Policy_, launch_policy), id,
                HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, ts)...);
        }

        template <typename Policy_, typename Client, typename Stub,
            typename Callback, typename... Ts>
        HPX_FORCEINLINE static hpx::future<
            typename traits::promise_local_result<typename traits::
                    extract_action<Action>::remote_result_type>::type>
        call(Policy_&& launch_policy,
            components::client_base<Client, Stub> const& c, Callback&& cb,
            Ts&&... ts)
        {
            typedef typename components::client_base<Client,
                Stub>::server_component_type component_type;

            typedef traits::is_valid_action<Action, component_type> is_valid;
            static_assert(is_valid::value,
                "The action to invoke is not supported by the target");

            return hpx::detail::async_cb_impl<Action>(
                HPX_FORWARD(Policy_, launch_policy), c.get_id(),
                HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, ts)...);
        }

        // distribution policy
        template <typename Policy_, typename DistPolicy, typename Callback,
            typename... Ts>
        HPX_FORCEINLINE static typename std::enable_if<
            traits::is_distribution_policy<DistPolicy>::value,
            hpx::future<typename traits::promise_local_result<typename traits::
                    extract_action<Action>::remote_result_type>::type>>::type
        call(Policy_&& launch_policy, DistPolicy const& policy, Callback&& cb,
            Ts&&... ts)
        {
            return policy.template async_cb<Action>(
                HPX_FORWARD(Policy_, launch_policy), HPX_FORWARD(Callback, cb),
                HPX_FORWARD(Ts, ts)...);
        }
    };

    // hpx::id_type
    template <typename Action>
    struct async_cb_action_dispatch<Action, hpx::id_type>
    {
        template <typename Callback, typename... Ts>
        HPX_FORCEINLINE static hpx::future<
            typename traits::promise_local_result<typename traits::
                    extract_action<Action>::remote_result_type>::type>
        call(hpx::id_type const& id, Callback&& cb, Ts&&... ts)
        {
            return async_cb_action_dispatch<Action,
                hpx::detail::async_policy>::call(launch::async, id,
                HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, ts)...);
        }
    };

    // component::client
    template <typename Action, typename Client>
    struct async_cb_action_dispatch<Action, Client,
        typename std::enable_if<traits::is_client<Client>::value>::type>
    {
        template <typename Client_, typename Stub, typename Callback,
            typename... Ts>
        HPX_FORCEINLINE static hpx::future<
            typename traits::promise_local_result<typename traits::
                    extract_action<Action>::remote_result_type>::type>
        call(components::client_base<Client_, Stub> const& c, Callback&& cb,
            Ts&&... ts)
        {
            typedef typename components::client_base<Client_,
                Stub>::server_component_type component_type;

            typedef traits::is_valid_action<Action, component_type> is_valid;
            static_assert(is_valid::value,
                "The action to invoke is not supported by the target");

            return async_cb_action_dispatch<Action,
                hpx::detail::async_policy>::call(launch::async, c.get_id(),
                HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, ts)...);
        }
    };

    // distribution policy
    template <typename Action, typename Policy>
    struct async_cb_action_dispatch<Action, Policy,
        typename std::enable_if<
            traits::is_distribution_policy<Policy>::value>::type>
    {
        template <typename DistPolicy, typename Callback, typename... Ts>
        HPX_FORCEINLINE static hpx::future<
            typename traits::promise_local_result<typename traits::
                    extract_action<Action>::remote_result_type>::type>
        call(DistPolicy const& policy, Callback&& cb, Ts&&... ts)
        {
            return async_cb_action_dispatch<Action,
                hpx::detail::async_policy>::call(launch::async, policy,
                HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, ts)...);
        }
    };
}}    // namespace hpx::detail

namespace hpx {
    template <typename Action, typename F, typename... Ts>
    HPX_FORCEINLINE auto async_cb(F&& f, Ts&&... ts)
        -> decltype(detail::async_cb_action_dispatch<Action,
            typename std::decay<F>::type>::call(HPX_FORWARD(F, f),
            HPX_FORWARD(Ts, ts)...))
    {
        return detail::async_cb_action_dispatch<Action,
            typename std::decay<F>::type>::call(HPX_FORWARD(F, f),
            HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace detail {
    // any action
    template <typename Action>
    struct async_cb_dispatch<Action,
        typename std::enable_if<traits::is_action<Action>::value>::type>
    {
        template <typename Component, typename Signature, typename Derived,
            typename Callback, typename... Ts>
        HPX_FORCEINLINE static hpx::future<
            typename traits::promise_local_result<typename traits::
                    extract_action<Derived>::remote_result_type>::type>
        call(hpx::actions::basic_action<Component, Signature, Derived> const&,
            hpx::id_type const& id, Callback&& cb, Ts&&... ts)
        {
            return async_cb<Derived>(
                id, HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, ts)...);
        }

        template <typename Component, typename Signature, typename Derived,
            typename Client, typename Stub, typename Callback, typename... Ts>
        HPX_FORCEINLINE static hpx::future<
            typename traits::promise_local_result<typename traits::
                    extract_action<Derived>::remote_result_type>::type>
        call(hpx::actions::basic_action<Component, Signature, Derived> const&,
            components::client_base<Client, Stub> const& c, Callback&& cb,
            Ts&&... ts)
        {
            typedef typename components::client_base<Client,
                Stub>::server_component_type component_type;

            typedef traits::is_valid_action<Derived, component_type> is_valid;
            static_assert(is_valid::value,
                "The action to invoke is not supported by the target");

            return async_cb<Derived>(
                c.get_id(), HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, ts)...);
        }

        template <typename Component, typename Signature, typename Derived,
            typename DistPolicy, typename Callback, typename... Ts>
        HPX_FORCEINLINE static hpx::future<
            typename traits::promise_local_result<typename traits::
                    extract_action<Derived>::remote_result_type>::type>
        call(hpx::actions::basic_action<Component, Signature, Derived> const&,
            DistPolicy const& policy, Callback&& cb, Ts&&... ts)
        {
            return async_cb<Derived>(
                policy, HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, ts)...);
        }
    };

    template <typename Policy>
    struct async_cb_dispatch<Policy,
        typename std::enable_if<traits::is_launch_policy<Policy>::value>::type>
    {
        template <typename Policy_, typename Component, typename Signature,
            typename Derived, typename Callback, typename... Ts>
        HPX_FORCEINLINE static hpx::future<
            typename traits::promise_local_result<typename traits::
                    extract_action<Derived>::remote_result_type>::type>
        call(Policy_&& launch_policy,
            hpx::actions::basic_action<Component, Signature, Derived> const&,
            hpx::id_type const& id, Callback&& cb, Ts&&... ts)
        {
            return async_cb<Derived>(HPX_FORWARD(Policy_, launch_policy), id,
                HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, ts)...);
        }

        template <typename Policy_, typename Component, typename Signature,
            typename Derived, typename Client, typename Stub, typename Callback,
            typename... Ts>
        HPX_FORCEINLINE static hpx::future<
            typename traits::promise_local_result<typename traits::
                    extract_action<Derived>::remote_result_type>::type>
        call(Policy_&& launch_policy,
            hpx::actions::basic_action<Component, Signature, Derived> const&,
            components::client_base<Client, Stub> const& c, Callback&& cb,
            Ts&&... ts)
        {
            typedef typename components::client_base<Client,
                Stub>::server_component_type component_type;

            typedef traits::is_valid_action<Derived, component_type> is_valid;
            static_assert(is_valid::value,
                "The action to invoke is not supported by the target");

            return async_cb<Derived>(HPX_FORWARD(Policy_, launch_policy),
                c.get_id(), HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, ts)...);
        }

        template <typename Policy_, typename Component, typename Signature,
            typename Derived, typename DistPolicy, typename Callback,
            typename... Ts>
        HPX_FORCEINLINE static hpx::future<
            typename traits::promise_local_result<typename traits::
                    extract_action<Derived>::remote_result_type>::type>
        call(Policy_&& launch_policy,
            hpx::actions::basic_action<Component, Signature, Derived> const&,
            DistPolicy const& policy, Callback&& cb, Ts&&... ts)
        {
            return async_cb<Derived>(HPX_FORWARD(Policy_, launch_policy),
                policy, HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, ts)...);
        }
    };
}}    // namespace hpx::detail

namespace hpx {
    template <typename F, typename... Ts>
    HPX_FORCEINLINE auto async_cb(F&& f, Ts&&... ts) -> decltype(
        detail::async_cb_dispatch<typename std::decay<F>::type>::call(
            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...))
    {
        return detail::async_cb_dispatch<typename std::decay<F>::type>::call(
            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx

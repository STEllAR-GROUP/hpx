//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_ASYNC_CALLBACK_MAR_30_2015_1119AM)
#define HPX_LCOS_ASYNC_CALLBACK_MAR_30_2015_1119AM

#include <hpx/config.hpp>
#include <hpx/lcos/async_callback_fwd.hpp>
#include <hpx/lcos/detail/async_implementations_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/traits/extract_action.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_client.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/traits/is_valid_action.hpp>
#include <hpx/traits/promise_local_result.hpp>

#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace detail
{
    // launch
    template <typename Action, typename Policy>
    struct async_cb_action_dispatch<Action, Policy,
        typename std::enable_if<
            traits::is_launch_policy<Policy>::value
        >::type>
    {
        // id_type
        template <typename Callback, typename ...Ts>
        HPX_FORCEINLINE static
        lcos::future<
            typename traits::promise_local_result<
                typename traits::extract_action<
                    Action
                >::remote_result_type
            >::type>
        call(launch launch_policy,
            naming::id_type const& id, Callback&& cb, Ts&&... ts)
        {
            return hpx::detail::async_cb_impl<Action>(launch_policy, id,
                std::forward<Callback>(cb), std::forward<Ts>(ts)...);
        }

        template <typename Client, typename Stub, typename Callback,
            typename ...Ts>
        HPX_FORCEINLINE static
        lcos::future<
            typename traits::promise_local_result<
                typename traits::extract_action<
                    Action
                >::remote_result_type
            >::type>
        call(launch launch_policy,
            components::client_base<Client, Stub> const& c, Callback&& cb,
            Ts&&... ts)
        {
            typedef typename components::client_base<
                    Client, Stub
                >::server_component_type component_type;

            typedef traits::is_valid_action<Action, component_type> is_valid;
            static_assert(is_valid::value,
                "The action to invoke is not supported by the target");

            return hpx::detail::async_cb_impl<Action>(launch_policy,
                c.get_id(), std::forward<Callback>(cb),
                std::forward<Ts>(ts)...);
        }

        // distribution policy
        template <typename DistPolicy, typename Callback, typename ...Ts>
        HPX_FORCEINLINE static
        typename std::enable_if<
            traits::is_distribution_policy<DistPolicy>::value,
            lcos::future<
                typename traits::promise_local_result<
                    typename traits::extract_action<
                        Action
                    >::remote_result_type
                >::type
            >
        >::type
        call(launch launch_policy,
            DistPolicy const& policy, Callback&& cb, Ts&&... ts)
        {
            return policy.template async_cb<Action>(launch_policy,
                std::forward<Callback>(cb), std::forward<Ts>(ts)...);
        }
    };

    // naming::id_type
    template <typename Action>
    struct async_cb_action_dispatch<Action, naming::id_type>
    {
        template <typename Callback, typename ...Ts>
        HPX_FORCEINLINE static
        lcos::future<
            typename traits::promise_local_result<
                typename traits::extract_action<
                    Action
                >::remote_result_type
            >::type>
        call(naming::id_type const& id, Callback&& cb, Ts&&... ts)
        {
            return async_cb_action_dispatch<
                    Action, launch
                >::call(launch::all, id, std::forward<Callback>(cb),
                    std::forward<Ts>(ts)...);
        }
    };

    // component::client
    template <typename Action, typename Client>
    struct async_cb_action_dispatch<Action, Client,
        typename std::enable_if<
            traits::is_client<Client>::value
        >::type>
    {
        template <typename Client_, typename Stub, typename Callback,
            typename ...Ts>
        HPX_FORCEINLINE static
        lcos::future<
            typename traits::promise_local_result<
                typename traits::extract_action<
                    Action
                >::remote_result_type
            >::type>
        call(components::client_base<Client_, Stub> const& c, Callback&& cb,
            Ts&&... ts)
        {
            typedef typename components::client_base<
                    Client_, Stub
                >::server_component_type component_type;

            typedef traits::is_valid_action<Action, component_type> is_valid;
            static_assert(is_valid::value,
                "The action to invoke is not supported by the target");

            return async_cb_action_dispatch<
                    Action, launch
                >::call(launch::all, c.get_id(), std::forward<Callback>(cb),
                    std::forward<Ts>(ts)...);
        }
    };

    // distribution policy
    template <typename Action, typename Policy>
    struct async_cb_action_dispatch<Action, Policy,
        typename std::enable_if<
            traits::is_distribution_policy<Policy>::value
        >::type>
    {
        template <typename DistPolicy, typename Callback, typename ...Ts>
        HPX_FORCEINLINE static
        lcos::future<
            typename traits::promise_local_result<
                typename traits::extract_action<
                    Action
                >::remote_result_type
            >::type>
        call(DistPolicy const& policy, Callback&& cb, Ts&&... ts)
        {
            return async_cb_action_dispatch<
                    Action, launch
                >::call(launch::all, policy, std::forward<Callback>(cb),
                    std::forward<Ts>(ts)...);
        }
    };
}}

namespace hpx
{
    template <typename Action, typename F, typename ...Ts>
    HPX_FORCEINLINE
    auto async_cb(F && f, Ts &&... ts)
    ->  decltype(detail::async_cb_action_dispatch<
                Action, typename util::decay<F>::type
            >::call(std::forward<F>(f), std::forward<Ts>(ts)...))
    {
        return detail::async_cb_action_dispatch<
                Action, typename util::decay<F>::type
            >::call(std::forward<F>(f), std::forward<Ts>(ts)...);
    }
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace detail
{
    // any action
    template <typename Action>
    struct async_cb_dispatch<Action,
        typename std::enable_if<
            traits::is_action<Action>::value
        >::type>
    {
        template <typename Component, typename Signature, typename Derived,
            typename Callback, typename ...Ts>
        HPX_FORCEINLINE static
        lcos::future<
            typename traits::promise_local_result<
                typename traits::extract_action<
                    Derived
                >::remote_result_type
            >::type>
        call(hpx::actions::basic_action<Component, Signature, Derived> const&,
            naming::id_type const& id, Callback&& cb, Ts&&... ts)
        {
            return async_cb<Derived>(id, std::forward<Callback>(cb),
                std::forward<Ts>(ts)...);
        }

        template <typename Component, typename Signature, typename Derived,
            typename Client, typename Stub, typename Callback, typename ...Ts>
        HPX_FORCEINLINE static
        lcos::future<
            typename traits::promise_local_result<
                typename traits::extract_action<
                    Derived
                >::remote_result_type
            >::type>
        call(hpx::actions::basic_action<Component, Signature, Derived> const&,
            components::client_base<Client, Stub> const& c, Callback&& cb,
            Ts&&... ts)
        {
            typedef typename components::client_base<
                    Client, Stub
                >::server_component_type component_type;

            typedef traits::is_valid_action<Derived, component_type> is_valid;
            static_assert(is_valid::value,
                "The action to invoke is not supported by the target");

            return async_cb<Derived>(c.get_id(), std::forward<Callback>(cb),
                std::forward<Ts>(ts)...);
        }

        template <typename Component, typename Signature, typename Derived,
            typename DistPolicy, typename Callback, typename ...Ts>
        HPX_FORCEINLINE static
        lcos::future<
            typename traits::promise_local_result<
                typename traits::extract_action<
                    Derived
                >::remote_result_type
            >::type>
        call(hpx::actions::basic_action<Component, Signature, Derived> const&,
            DistPolicy const& policy, Callback&& cb, Ts&&... ts)
        {
            return async_cb<Derived>(policy, std::forward<Callback>(cb),
                std::forward<Ts>(ts)...);
        }
    };

    template <typename Policy>
    struct async_cb_dispatch<Policy,
        typename std::enable_if<
            traits::is_launch_policy<Policy>::value
        >::type>
    {
        template <typename Component, typename Signature, typename Derived,
            typename Callback, typename ...Ts>
        HPX_FORCEINLINE static
        lcos::future<
            typename traits::promise_local_result<
                typename traits::extract_action<
                    Derived
                >::remote_result_type
            >::type>
        call(launch launch_policy,
            hpx::actions::basic_action<Component, Signature, Derived> const&,
            naming::id_type const& id, Callback&& cb, Ts&&... ts)
        {
            return async_cb<Derived>(launch_policy, id,
                std::forward<Callback>(cb), std::forward<Ts>(ts)...);
        }

        template <typename Component, typename Signature, typename Derived,
            typename Client, typename Stub, typename Callback, typename ...Ts>
        HPX_FORCEINLINE static
        lcos::future<
            typename traits::promise_local_result<
                typename traits::extract_action<
                    Derived
                >::remote_result_type
            >::type>
        call(launch launch_policy,
            hpx::actions::basic_action<Component, Signature, Derived> const&,
            components::client_base<Client, Stub> const& c, Callback&& cb,
            Ts&&... ts)
        {
            typedef typename components::client_base<
                    Client, Stub
                >::server_component_type component_type;

            typedef traits::is_valid_action<Derived, component_type> is_valid;
            static_assert(is_valid::value,
                "The action to invoke is not supported by the target");

            return async_cb<Derived>(launch_policy, c.get_id(),
                std::forward<Callback>(cb), std::forward<Ts>(ts)...);
        }

        template <typename Component, typename Signature, typename Derived,
            typename DistPolicy, typename Callback, typename ...Ts>
        HPX_FORCEINLINE static
        lcos::future<
            typename traits::promise_local_result<
                typename traits::extract_action<
                    Derived
                >::remote_result_type
            >::type>
        call(launch launch_policy,
            hpx::actions::basic_action<Component, Signature, Derived> const&,
            DistPolicy const& policy, Callback&& cb, Ts&&... ts)
        {
            return async_cb<Derived>(launch_policy, policy,
                std::forward<Callback>(cb), std::forward<Ts>(ts)...);
        }
    };
}}

namespace hpx
{
    template <typename F, typename ...Ts>
    HPX_FORCEINLINE
    auto async_cb(F && f, Ts &&... ts)
    ->  decltype(detail::async_cb_dispatch<
                typename util::decay<F>::type
            >::call(std::forward<F>(f), std::forward<Ts>(ts)...))
    {
        return detail::async_cb_dispatch<
                typename util::decay<F>::type
            >::call(std::forward<F>(f), std::forward<Ts>(ts)...);
    }
}

#endif


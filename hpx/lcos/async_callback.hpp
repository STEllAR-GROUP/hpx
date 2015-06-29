//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_ASYNC_CALLBACK_MAR_30_2015_1119AM)
#define HPX_LCOS_ASYNC_CALLBACK_MAR_30_2015_1119AM

#include <hpx/config.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/lcos/detail/async_implementations_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/async_callback_fwd.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/static_assert.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace detail
{
    // BOOST_SCOPED_ENUM(launch)
    template <typename Action, typename Policy>
    struct async_cb_action_dispatch<Action, Policy,
        typename boost::enable_if_c<
            traits::is_launch_policy<Policy>::value
        >::type>
    async_cb(launch policy, naming::id_type const& gid,
        Callback&& cb, Ts&&... vs)
    {
        // id_type
        template <typename Callback, typename ...Ts>
        BOOST_FORCEINLINE static
        lcos::future<
            typename traits::promise_local_result<
                typename traits::extract_action<
                    Action
                >::remote_result_type
            >::type>
        call(BOOST_SCOPED_ENUM(launch) launch_policy,
            naming::id_type const& id, Callback&& cb, Ts&&... ts)
        {
            return hpx::detail::async_cb_impl<Action>(launch_policy, id,
                std::forward<Callback>(cb), std::forward<Ts>(ts)...);
        }

        template <typename Client, typename Stub, typename Callback,
            typename ...Ts>
        BOOST_FORCEINLINE static
        lcos::future<
            typename traits::promise_local_result<
                typename traits::extract_action<
                    Action
                >::remote_result_type
            >::type>
        call(BOOST_SCOPED_ENUM(launch) launch_policy,
            components::client_base<Client, Stub> const& c, Callback&& cb,
            Ts&&... ts)
        {
            typedef typename components::client_base<
                    Client, Stub
                >::server_component_type component_type;
            BOOST_STATIC_ASSERT(
                traits::is_valid_action<Action, component_type>::value
            );

            return hpx::detail::async_cb_impl<Action>(launch_policy,
                c.get_id(), std::forward<Callback>(cb),
                std::forward<Ts>(ts)...);
        }

        // distribution policy
        template <typename DistPolicy, typename Callback, typename ...Ts>
        BOOST_FORCEINLINE static
        typename boost::enable_if_c<
            traits::is_distribution_policy<DistPolicy>::value,
            lcos::future<
                typename traits::promise_local_result<
                    typename traits::extract_action<
                        Action
                    >::remote_result_type
                >::type
            >
        >::type
        call(BOOST_SCOPED_ENUM(launch) launch_policy,
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
        BOOST_FORCEINLINE static
        lcos::future<
            typename traits::promise_local_result<
                typename traits::extract_action<
                    Action
                >::remote_result_type
            >::type>
        call(naming::id_type const& id, Callback&& cb, Ts&&... ts)
        {
            return async_cb_action_dispatch<
                    Action, BOOST_SCOPED_ENUM(launch)
                >::call(launch::all, id, std::forward<Callback>(cb),
                    std::forward<Ts>(ts)...);
        }
    };

    // component::client
    template <typename Action, typename Client>
    struct async_cb_action_dispatch<Action, Client,
        typename boost::enable_if_c<
            traits::is_client<Client>::value
        >::type>
    {
        template <typename Client_, typename Stub, typename Callback,
            typename ...Ts>
        BOOST_FORCEINLINE static
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
            BOOST_STATIC_ASSERT(
                traits::is_valid_action<Action, component_type>::value
            );

            return async_cb_action_dispatch<
                    Action, BOOST_SCOPED_ENUM(launch)
                >::call(launch::all, c.get_id(), std::forward<Callback>(cb),
                    std::forward<Ts>(ts)...);
        }
    };

    // distribution policy
    template <typename Action, typename Policy>
    struct async_cb_action_dispatch<Action, Policy,
        typename boost::enable_if_c<
            traits::is_distribution_policy<Policy>::value
        >::type>
    async_cb(launch policy,
        hpx::actions::basic_action<Component, Signature, Derived> const& /*act*/,
        naming::id_type const& gid, Callback&& cb, Ts&&... vs)
    {
        template <typename DistPolicy, typename Callback, typename ...Ts>
        BOOST_FORCEINLINE static
        lcos::future<
            typename traits::promise_local_result<
                typename traits::extract_action<
                    Action
                >::remote_result_type
            >::type>
        call(DistPolicy const& policy, Callback&& cb, Ts&&... ts)
        {
            return async_cb_action_dispatch<
                    Action, BOOST_SCOPED_ENUM(launch)
                >::call(launch::all, policy, std::forward<Callback>(cb),
                    std::forward<Ts>(ts)...);
        }
    };
}}

namespace hpx
{
    template <typename Action, typename F, typename ...Ts>
    BOOST_FORCEINLINE
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
        typename boost::enable_if_c<
            traits::is_action<Action>::value
        >::type>
    {
        template <typename Component, typename Signature, typename Derived,
            typename Callback, typename ...Ts>
        BOOST_FORCEINLINE static
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
        BOOST_FORCEINLINE static
        lcos::future<
            typename traits::promise_local_result<
                typename traits::extract_action<
                    Derived
                >::remote_result_type
            >::type>
    >::type
    async_cb(launch launch_policy, DistPolicy const& policy,
        Callback&& cb, Ts&&... vs)
    {
        return policy.template async_cb<Action>(launch_policy,
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

            return async_cb<Derived>(c.get_id(), std::forward<Callback>(cb),
                std::forward<Ts>(ts)...);
        }

        template <typename Component, typename Signature, typename Derived,
            typename DistPolicy, typename Callback, typename ...Ts>
        BOOST_FORCEINLINE static
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
        typename boost::enable_if_c<
            traits::is_launch_policy<Policy>::value
        >::type>
    {
        template <typename Component, typename Signature, typename Derived,
            typename Callback, typename ...Ts>
        BOOST_FORCEINLINE static
        lcos::future<
            typename traits::promise_local_result<
                typename traits::extract_action<
                    Derived
                >::remote_result_type
            >::type>
        call(BOOST_SCOPED_ENUM(launch) launch_policy,
            hpx::actions::basic_action<Component, Signature, Derived> const&,
            naming::id_type const& id, Callback&& cb, Ts&&... ts)
        {
            return async_cb<Derived>(launch_policy, id,
                std::forward<Callback>(cb), std::forward<Ts>(ts)...);
        }

        template <typename Component, typename Signature, typename Derived,
            typename Client, typename Stub, typename Callback, typename ...Ts>
        BOOST_FORCEINLINE static
        lcos::future<
            typename traits::promise_local_result<
                typename traits::extract_action<
                    Derived
                >::remote_result_type
            >::type>
    >::type
    async_cb(launch launch_policy,
        hpx::actions::basic_action<Component, Signature, Derived> const& /*act*/,
        DistPolicy const& policy, Callback&& cb, Ts&&... vs)
    {
        return async_cb<Derived>(launch_policy, policy,
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

            return async_cb<Derived>(launch_policy, c.get_id(),
                std::forward<Callback>(cb), std::forward<Ts>(ts)...);
        }

        template <typename Component, typename Signature, typename Derived,
            typename DistPolicy, typename Callback, typename ...Ts>
        BOOST_FORCEINLINE static
        lcos::future<
            typename traits::promise_local_result<
                typename traits::extract_action<
                    Derived
                >::remote_result_type
            >::type>
        call(BOOST_SCOPED_ENUM(launch) launch_policy,
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
    BOOST_FORCEINLINE
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

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
#include <hpx/lcos/async.hpp>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async_cb(launch policy, naming::id_type const& gid,
        Callback&& cb, Ts&&... vs)
    {
        return hpx::detail::async_cb_impl<Action>(policy, gid,
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Callback, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async_cb(naming::id_type const& gid, Callback&& cb, Ts&&... vs)
    {
        return async_cb<Action>(launch::all, gid, std::forward<Callback>(cb),
            std::forward<Ts>(vs)...);
    }

    template <
        typename Component, typename Signature, typename Derived,
        typename Callback, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async_cb(launch policy,
        hpx::actions::basic_action<Component, Signature, Derived> const& /*act*/,
        naming::id_type const& gid, Callback&& cb, Ts&&... vs)
    {
        return async_cb<Derived>(policy, gid, std::forward<Callback>(cb),
            std::forward<Ts>(vs)...);
    }

    template <
        typename Component, typename Signature, typename Derived,
        typename Callback, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async_cb(
        hpx::actions::basic_action<Component, Signature, Derived> const& /*act*/,
        naming::id_type const& gid, Callback&& cb, Ts&&... vs)
    {
        return async_cb<Derived>(launch::all, gid, std::forward<Callback>(cb),
            std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename DistPolicy, typename Callback,
        typename ...Ts>
    typename boost::enable_if_c<
        traits::is_distribution_policy<DistPolicy>::value,
        lcos::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::remote_result_type
            >::type>
    >::type
    async_cb(launch launch_policy, DistPolicy const& policy,
        Callback&& cb, Ts&&... vs)
    {
        return policy.template async_cb<Action>(launch_policy,
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Action, typename DistPolicy, typename Callback,
        typename ...Ts>
    typename boost::enable_if_c<
        traits::is_distribution_policy<DistPolicy>::value,
        lcos::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::remote_result_type
            >::type>
    >::type
    async_cb(DistPolicy const& policy, Callback&& cb, Ts&&... vs)
    {
        return async_cb<Action>(launch::all, policy,
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <
        typename Component, typename Signature, typename Derived,
        typename DistPolicy, typename Callback, typename ...Ts>
    typename boost::enable_if_c<
        traits::is_distribution_policy<DistPolicy>::value,
        lcos::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<Derived>::remote_result_type
            >::type>
    >::type
    async_cb(launch launch_policy,
        hpx::actions::basic_action<Component, Signature, Derived> const& /*act*/,
        DistPolicy const& policy, Callback&& cb, Ts&&... vs)
    {
        return async_cb<Derived>(launch_policy, policy,
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <
        typename Component, typename Signature, typename Derived,
        typename DistPolicy, typename Callback, typename ...Ts>
    typename boost::enable_if_c<
        traits::is_distribution_policy<DistPolicy>::value,
        lcos::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<Derived>::remote_result_type
            >::type>
    >::type
    async_cb(
        hpx::actions::basic_action<Component, Signature, Derived> const& /*act*/,
        DistPolicy const& policy, Callback&& cb, Ts&&... vs)
    {
        return async_cb<Derived>(launch::all, policy,
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }
}

#endif

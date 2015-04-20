//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_ASYNC_SEP_28_2011_0840AM)
#define HPX_LCOS_ASYNC_SEP_28_2011_0840AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/lcos/detail/async_implementations.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/async_fwd.hpp>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async(BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& id,
        Ts&&... vs)
    {
        return hpx::detail::async_impl<Action>(policy, id,
            std::forward<Ts>(vs)...);
    }

    template <typename Action, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async(naming::id_type const& id, Ts&&... vs)
    {
        return async<Action>(launch::all, id, std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename Signature, typename Derived,
        typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async(BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::basic_action<Component, Signature, Derived> const& /*act*/,
        naming::id_type const& id, Ts&&... vs)
    {
        return async<Derived>(policy, id, std::forward<Ts>(vs)...);
    }

    template <
        typename Component, typename Signature, typename Derived,
        typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async(
        hpx::actions::basic_action<Component, Signature, Derived> const& /*act*/,
        naming::id_type const& id, Ts&&... vs)
    {
        return async<Derived>(launch::all, id, std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename DistPolicy, typename ...Ts>
    typename boost::enable_if_c<
        traits::is_distribution_policy<DistPolicy>::value,
        lcos::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::remote_result_type
            >::type>
    >::type
    async(BOOST_SCOPED_ENUM(launch) launch_policy, DistPolicy const& policy,
        Ts&&... vs)
    {
        return policy.template async<Action>(launch_policy,
            std::forward<Ts>(vs)...);
    }

    template <typename Action, typename DistPolicy, typename ...Ts>
    typename boost::enable_if_c<
        traits::is_distribution_policy<DistPolicy>::value,
        lcos::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::remote_result_type
            >::type>
    >::type
    async(DistPolicy const& policy, Ts&&... vs)
    {
        return async<Action>(launch::all, policy, std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename Signature, typename Derived,
        typename DistPolicy, typename ...Ts>
    typename boost::enable_if_c<
        traits::is_distribution_policy<DistPolicy>::value,
        lcos::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<Derived>::remote_result_type
            >::type>
    >::type
    async(BOOST_SCOPED_ENUM(launch) launch_policy,
        hpx::actions::basic_action<Component, Signature, Derived> const& /*act*/,
        DistPolicy const& policy, Ts&&... vs)
    {
        return async<Derived>(launch_policy, policy, std::forward<Ts>(vs)...);
    }

    template <
        typename Component, typename Signature, typename Derived,
        typename DistPolicy, typename ...Ts>
    typename boost::enable_if_c<
        traits::is_distribution_policy<DistPolicy>::value,
        lcos::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<Derived>::remote_result_type
            >::type>
    >::type
    async(
        hpx::actions::basic_action<Component, Signature, Derived> const& /*act*/,
        DistPolicy const& policy, Ts&&... vs)
    {
        return async<Derived>(launch::all, policy, std::forward<Ts>(vs)...);
    }
}

#endif

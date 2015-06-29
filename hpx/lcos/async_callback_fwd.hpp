//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_ASYNC_CALLBACK_FWD_MAR_30_2015_1122AM)
#define HPX_LCOS_ASYNC_CALLBACK_FWD_MAR_30_2015_1122AM

#include <hpx/traits.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/actions/basic_action_fwd.hpp>
#include <hpx/lcos/async_fwd.hpp>
#include <hpx/lcos/future.hpp>

#ifndef HPX_MSVC
#include <boost/utility/enable_if.hpp>
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async_cb(launch policy, naming::id_type const& gid,
        Callback&& cb, Ts&&... vs);

        // dispatch point used for async_cb<Action> implementations
        template <typename Action, typename Func, typename Enable = void>
        struct async_cb_action_dispatch;
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
        naming::id_type const& gid, Callback&& cb, Ts&&... vs);

    template <
        typename Component, typename Signature, typename Derived,
        typename Callback, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async_cb(launch policy,
        hpx::actions::basic_action<Component, Signature, Derived> const& /*act*/,
        naming::id_type const& gid, Callback&& cb, Ts&&... vs);

    ///////////////////////////////////////////////////////////////////////////
    // MSVC complains about ambiguities if it sees this forward declaration
#ifndef HPX_MSVC
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
        Callback&& cb, Ts&&... vs);

    template <typename Action, typename DistPolicy, typename Callback,
        typename ...Ts>
    typename boost::enable_if_c<
        traits::is_distribution_policy<DistPolicy>::value,
        lcos::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::remote_result_type
            >::type>
    >::type
    async_cb(DistPolicy const& policy, Callback&& cb, Ts&&... vs);

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
        DistPolicy const& policy, Callback&& cb, Ts&&... vs);

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
        DistPolicy const& policy, Callback&& cb, Ts&&... vs);
#endif
}

#endif

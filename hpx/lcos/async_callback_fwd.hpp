//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_ASYNC_CALLBACK_FWD_MAR_30_2015_1122AM)
#define HPX_LCOS_ASYNC_CALLBACK_FWD_MAR_30_2015_1122AM

#include <hpx/lcos/async_fwd.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async_cb(BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid,
        Callback&& cb, Ts&&... vs);

    template <typename Action, typename Callback, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async_cb(naming::id_type const& gid, Callback&& cb, Ts&&... vs);

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
    async_cb(BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::basic_action<Component, Signature, Derived> const& /*act*/,
        naming::id_type const& gid, Callback&& cb, Ts&&... vs);
}

#endif

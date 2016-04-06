//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_ASYNC_CONTINUE_CALLBACK_FWD_MAR_30_2015_1130AM)
#define HPX_LCOS_ASYNC_CONTINUE_CALLBACK_FWD_MAR_30_2015_1130AM

#include <hpx/lcos/async_continue_fwd.hpp>
#ifndef HPX_MSVC
#include <boost/utility/enable_if.hpp>
#endif

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <
            typename Action, typename RemoteResult, typename Cont,
            typename Target, typename Callback, typename ...Ts>
        lcos::future<
            typename traits::promise_local_result<
                typename result_of_async_continue<Action, Cont>::type
            >::type
        >
        async_continue_r_cb(Cont&& cont, Target const& target,
            Callback&& cb, Ts&&... vs);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Cont, typename Callback, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename detail::result_of_async_continue<Action, Cont>::type
        >::type
    >
    async_continue_cb(Cont&& cont, naming::id_type const& gid, Callback&& cb,
        Ts&&... vs);

    template <
        typename Component, typename Signature, typename Derived,
        typename Cont, typename Callback, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename detail::result_of_async_continue<Derived, Cont>::type
        >::type
    >
    async_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/
      , Cont&& cont, naming::id_type const& gid, Callback&& cb, Ts&&... vs);

    ///////////////////////////////////////////////////////////////////////////
    // MSVC complains about ambiguities if it sees this forward declaration
#ifndef HPX_MSVC
    template <typename Action, typename Cont, typename DistPolicy,
        typename Callback, typename ...Ts>
    typename boost::enable_if_c<
        traits::is_distribution_policy<DistPolicy>::value,
        lcos::future<
            typename traits::promise_local_result<
                typename detail::result_of_async_continue<Action, Cont>::type
            >::type>
    >::type
    async_continue_cb(Cont&& cont, DistPolicy const& policy, Callback&& cb,
        Ts&&... vs);

    template <
        typename Component, typename Signature, typename Derived,
        typename Cont, typename DistPolicy, typename Callback, typename ...Ts>
    typename boost::enable_if_c<
        traits::is_distribution_policy<DistPolicy>::value,
        lcos::future<
            typename traits::promise_local_result<
                typename detail::result_of_async_continue<Derived, Cont>::type
            >::type>
    >::type
    async_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/
      , Cont&& cont, DistPolicy const& policy, Callback&& cb, Ts&&... vs);
#endif
}

#endif

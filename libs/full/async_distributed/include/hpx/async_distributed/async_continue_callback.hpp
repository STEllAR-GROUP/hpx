//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/actions_base/traits/is_distribution_policy.hpp>
#include <hpx/async_distributed/async_callback_fwd.hpp>
#include <hpx/async_distributed/async_continue.hpp>
#include <hpx/async_distributed/detail/post_callback.hpp>
#include <hpx/futures/traits/promise_local_result.hpp>
#include <hpx/futures/traits/promise_remote_result.hpp>

#include <type_traits>
#include <utility>

namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Action, typename RemoteResult, typename Cont,
            typename Target, typename Callback, typename... Ts>
        hpx::future<typename traits::promise_local_result<
            typename result_of_async_continue<Action, Cont>::type>::type>
        async_continue_r_cb(
            Cont&& cont, Target const& target, Callback&& cb, Ts&&... vs)
        {
            typedef typename traits::promise_local_result<
                typename result_of_async_continue<Action, Cont>::type>::type
                result_type;

            typedef
                typename hpx::traits::extract_action<Action>::remote_result_type
                    continuation_result_type;

            hpx::distributed::promise<result_type, RemoteResult> p;
            auto f = p.get_future();

            hpx::post_cb<Action>(hpx::actions::typed_continuation<result_type,
                                     continuation_result_type>(
                                     p.get_id(), HPX_FORWARD(Cont, cont)),
                target, HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);

            return f;
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Cont, typename Callback, typename... Ts>
    hpx::future<typename traits::promise_local_result<
        typename detail::result_of_async_continue<Action, Cont>::type>::type>
    async_continue_cb(
        Cont&& cont, hpx::id_type const& gid, Callback&& cb, Ts&&... vs)
    {
        typedef typename traits::promise_remote_result<
            typename detail::result_of_async_continue<Action, Cont>::type>::type
            result_type;

        return detail::async_continue_r_cb<Action, result_type>(
            HPX_FORWARD(Cont, cont), gid, HPX_FORWARD(Callback, cb),
            HPX_FORWARD(Ts, vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename Cont, typename Callback, typename... Ts>
    hpx::future<typename traits::promise_local_result<
        typename detail::result_of_async_continue<Derived, Cont>::type>::type>
    async_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        Cont&& cont, hpx::id_type const& gid, Callback&& cb, Ts&&... vs)
    {
        return async_continue_cb<Derived>(HPX_FORWARD(Cont, cont), gid,
            HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Cont, typename DistPolicy,
        typename Callback, typename... Ts>
    std::enable_if_t<traits::is_distribution_policy_v<DistPolicy>,
        hpx::future<typename traits::promise_local_result<typename detail::
                result_of_async_continue<Action, Cont>::type>::type>>
    async_continue_cb(
        Cont&& cont, DistPolicy const& policy, Callback&& cb, Ts&&... vs)
    {
        typedef typename traits::promise_remote_result<
            typename detail::result_of_async_continue<Action, Cont>::type>::type
            result_type;

        return detail::async_continue_r_cb<Action, result_type>(
            HPX_FORWARD(Cont, cont), policy, HPX_FORWARD(Callback, cb),
            HPX_FORWARD(Ts, vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename Cont, typename DistPolicy, typename Callback, typename... Ts>
    std::enable_if_t<traits::is_distribution_policy_v<DistPolicy>,
        hpx::future<typename traits::promise_local_result<typename detail::
                result_of_async_continue<Derived, Cont>::type>::type>>
    async_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        Cont&& cont, DistPolicy const& policy, Callback&& cb, Ts&&... vs)
    {
        return async_continue_cb<Derived>(HPX_FORWARD(Cont, cont), policy,
            HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
    }
}    // namespace hpx

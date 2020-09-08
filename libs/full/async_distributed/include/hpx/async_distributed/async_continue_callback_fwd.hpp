//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/async_distributed/async_continue_fwd.hpp>
#include <hpx/futures/traits/promise_local_result.hpp>
#include <hpx/traits/is_distribution_policy.hpp>

#ifndef HPX_MSVC
#include <type_traits>
#endif

namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Action, typename RemoteResult, typename Cont,
            typename Target, typename Callback, typename... Ts>
        lcos::future<typename traits::promise_local_result<
            typename result_of_async_continue<Action, Cont>::type>::type>
        async_continue_r_cb(
            Cont&& cont, Target const& target, Callback&& cb, Ts&&... vs);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Cont, typename Callback, typename... Ts>
    lcos::future<typename traits::promise_local_result<
        typename detail::result_of_async_continue<Action, Cont>::type>::type>
    async_continue_cb(
        Cont&& cont, naming::id_type const& gid, Callback&& cb, Ts&&... vs);

    template <typename Component, typename Signature, typename Derived,
        typename Cont, typename Callback, typename... Ts>
    lcos::future<typename traits::promise_local_result<
        typename detail::result_of_async_continue<Derived, Cont>::type>::type>
    async_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/
        ,
        Cont&& cont, naming::id_type const& gid, Callback&& cb, Ts&&... vs);

    ///////////////////////////////////////////////////////////////////////////
    // MSVC complains about ambiguities if it sees this forward declaration
#ifndef HPX_MSVC
    template <typename Action, typename Cont, typename DistPolicy,
        typename Callback, typename... Ts>
    typename std::enable_if<traits::is_distribution_policy<DistPolicy>::value,
        lcos::future<typename traits::promise_local_result<typename detail::
                result_of_async_continue<Action, Cont>::type>::type>>::type
    async_continue_cb(
        Cont&& cont, DistPolicy const& policy, Callback&& cb, Ts&&... vs);

    template <typename Component, typename Signature, typename Derived,
        typename Cont, typename DistPolicy, typename Callback, typename... Ts>
    typename std::enable_if<traits::is_distribution_policy<DistPolicy>::value,
        lcos::future<typename traits::promise_local_result<typename detail::
                result_of_async_continue<Derived, Cont>::type>::type>>::type
    async_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/
        ,
        Cont&& cont, DistPolicy const& policy, Callback&& cb, Ts&&... vs);
#endif
}    // namespace hpx

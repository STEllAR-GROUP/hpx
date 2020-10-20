//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/basic_action_fwd.hpp>
#include <hpx/actions_base/traits/action_remote_result.hpp>
#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/promise_local_result.hpp>
#include <hpx/modules/naming.hpp>
#include <hpx/traits/is_distribution_policy.hpp>

#include <type_traits>

namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Action, typename Cont>
        struct result_of_async_continue
          : traits::action_remote_result<typename util::invoke_result<
                typename std::decay<Cont>::type, naming::id_type,
                typename hpx::traits::extract_action<
                    Action>::remote_result_type>::type>
        {
        };

        template <typename Action, typename RemoteResult, typename Cont,
            typename Target, typename... Ts>
        lcos::future<typename traits::promise_local_result<
            typename result_of_async_continue<Action, Cont>::type>::type>
        async_continue_r(Cont&& cont, Target const& target, Ts&&... vs);
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Cont, typename... Ts>
    lcos::future<typename traits::promise_local_result<
        typename detail::result_of_async_continue<Action, Cont>::type>::type>
    async_continue(Cont&& cont, naming::id_type const& gid, Ts&&... vs);

    template <typename Component, typename Signature, typename Derived,
        typename Cont, typename... Ts>
    lcos::future<typename traits::promise_local_result<
        typename detail::result_of_async_continue<Derived, Cont>::type>::type>
    async_continue(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/
        ,
        Cont&& cont, naming::id_type const& gid, Ts&&... vs);

    ///////////////////////////////////////////////////////////////////////////
    // MSVC complains about ambiguities if it sees this forward declaration
#ifndef HPX_MSVC
    template <typename Action, typename Cont, typename DistPolicy,
        typename... Ts>
    typename std::enable_if<traits::is_distribution_policy<DistPolicy>::value,
        lcos::future<typename traits::promise_local_result<typename detail::
                result_of_async_continue<Action, Cont>::type>::type>>::type
    async_continue(Cont&& cont, DistPolicy const& policy, Ts&&... vs);

    template <typename Component, typename Signature, typename Derived,
        typename Cont, typename DistPolicy, typename... Ts>
    typename std::enable_if<traits::is_distribution_policy<DistPolicy>::value,
        lcos::future<typename traits::promise_local_result<typename detail::
                result_of_async_continue<Derived, Cont>::type>::type>>::type
    async_continue(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/
        ,
        Cont&& cont, DistPolicy const& policy, Ts&&... vs);
#endif
}    // namespace hpx

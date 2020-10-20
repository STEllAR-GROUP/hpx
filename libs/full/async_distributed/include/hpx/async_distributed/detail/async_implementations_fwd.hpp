//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_local/async_fwd.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/naming.hpp>

namespace hpx { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Launch, typename... Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::type::local_result_type>
    async_impl(Launch&& policy, hpx::id_type const& id, Ts&&... vs);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback, typename... Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::type::local_result_type>
    async_cb_impl(
        launch policy, hpx::id_type const& id, Callback&& cb, Ts&&... vs);

    template <typename Action, typename Callback, typename... Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::type::local_result_type>
    async_cb_impl(hpx::detail::sync_policy, hpx::id_type const& id,
        Callback&& cb, Ts&&... vs);

    template <typename Action, typename Callback, typename... Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::type::local_result_type>
    async_cb_impl(hpx::detail::async_policy, hpx::id_type const& id,
        Callback&& cb, Ts&&... vs);

    template <typename Action, typename Callback, typename... Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::type::local_result_type>
    async_cb_impl(hpx::detail::deferred_policy, hpx::id_type const& id,
        Callback&& cb, Ts&&... vs);
}}    // namespace hpx::detail

//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_ASYNC_IMPLEMENTATIONS_FWD_APR_13_2015_0829AM)
#define HPX_LCOS_ASYNC_IMPLEMENTATIONS_FWD_APR_13_2015_0829AM

#include <hpx/config.hpp>
#include <hpx/lcos/async_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/traits/extract_action.hpp>

namespace hpx { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename ...Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::type::local_result_type
    >
    async_impl(launch policy, hpx::id_type const& id,
        Ts&&... vs);

    template <typename Action, typename ...Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::type::local_result_type
    >
    async_impl(hpx::detail::sync_policy, hpx::id_type const& id,
        Ts&&... vs);

    template <typename Action, typename ...Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::type::local_result_type
    >
    async_impl(hpx::detail::async_policy, hpx::id_type const& id,
        Ts&&... vs);

    template <typename Action, typename ...Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::type::local_result_type
    >
    async_impl(hpx::detail::deferred_policy, hpx::id_type const& id,
        Ts&&... vs);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback, typename ...Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::type::local_result_type
    >
    async_cb_impl(launch policy, hpx::id_type const& id,
        Callback&& cb, Ts&&... vs);

    template <typename Action, typename Callback, typename ...Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::type::local_result_type
    >
    async_cb_impl(hpx::detail::sync_policy, hpx::id_type const& id,
        Callback&& cb, Ts&&... vs);

    template <typename Action, typename Callback, typename ...Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::type::local_result_type
    >
    async_cb_impl(hpx::detail::async_policy, hpx::id_type const& id,
        Callback&& cb, Ts&&... vs);

    template <typename Action, typename Callback, typename ...Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::type::local_result_type
    >
    async_cb_impl(hpx::detail::deferred_policy, hpx::id_type const& id,
        Callback&& cb, Ts&&... vs);
}}

#endif

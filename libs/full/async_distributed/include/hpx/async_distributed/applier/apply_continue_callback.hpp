//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/actions_base_support.hpp>
#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/async_distributed/applier/apply.hpp>
#include <hpx/async_distributed/applier/apply_callback.hpp>
#include <hpx/async_distributed/make_continuation.hpp>

#include <utility>

namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Cont, typename Callback, typename... Ts>
    bool apply_continue_cb(
        Cont&& cont, id_type const& gid, Callback&& cb, Ts&&... vs)
    {
        using action_type = typename hpx::traits::extract_action<Action>::type;
        using result_type = typename action_type::remote_result_type;
        using local_result_type = typename action_type::local_result_type;

        return apply_cb<Action>(
            hpx::actions::typed_continuation<local_result_type, result_type>(
                std::forward<Cont>(cont)),
            gid, std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename Cont, typename Callback, typename... Ts>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        Cont&& cont, id_type const& gid, Callback&& cb, Ts&&... vs)
    {
        return apply_continue_cb<Derived>(std::forward<Cont>(cont), gid,
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback, typename... Ts>
    bool apply_continue_cb(
        id_type const& cont, id_type const& gid, Callback&& cb, Ts&&... vs)
    {
        using action_type = typename hpx::traits::extract_action<Action>::type;
        using result_type = typename action_type::remote_result_type;
        using local_result_type = typename action_type::local_result_type;

        return apply_cb<Action>(
            hpx::actions::typed_continuation<local_result_type, result_type>(
                cont, make_continuation()),
            gid, std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename Callback, typename... Ts>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        id_type const& cont, id_type const& gid, Callback&& cb, Ts&&... vs)
    {
        return apply_continue_cb<Derived>(
            cont, gid, std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }
}    // namespace hpx

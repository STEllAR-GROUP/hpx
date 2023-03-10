//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/actions_base/actions_base_support.hpp>
#include <hpx/actions_base/traits/action_priority.hpp>
#include <hpx/agas_base/primary_namespace.hpp>
#include <hpx/agas_base/server/primary_namespace.hpp>
#include <hpx/async_colocated/functional/colocated_helpers.hpp>
#include <hpx/async_colocated/post_colocated_callback_fwd.hpp>
#include <hpx/async_colocated/register_post_colocated.hpp>
#include <hpx/async_distributed/bind_action.hpp>
#include <hpx/async_distributed/detail/post_continue_callback.hpp>
#include <hpx/functional/bind.hpp>

#include <utility>

namespace hpx::detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback, typename... Ts>
    bool post_colocated_cb(hpx::id_type const& gid, Callback&& cb, Ts&&... vs)
    {
        // shortcut co-location code if target already is a locality
        if (naming::is_locality(gid))
        {
            return hpx::post_cb<Action>(
                gid, HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
        }

        // Attach the requested action as a continuation to a resolve_async
        // call on the locality responsible for the target gid.
        hpx::id_type service_target(
            agas::primary_namespace::get_service_instance(gid.get_gid()),
            hpx::id_type::management_type::unmanaged);

        using action_type = agas::server::primary_namespace::colocate_action;

        using placeholders::_2;
        return post_continue_cb<action_type>(
            util::functional::post_continuation(hpx::bind<Action>(
                hpx::bind(util::functional::extract_locality(), _2, gid),
                HPX_FORWARD(Ts, vs)...)),
            service_target, HPX_FORWARD(Callback, cb), gid.get_gid());
    }

    template <typename Component, typename Signature, typename Derived,
        typename Callback, typename... Ts>
    bool post_colocated_cb(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        hpx::id_type const& gid, Callback&& cb, Ts&&... vs)
    {
        return post_colocated_cb<Derived>(
            gid, HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Continuation, typename Callback,
        typename... Ts>
    bool post_colocated_cb(
        Continuation&& cont, hpx::id_type const& gid, Callback&& cb, Ts&&... vs)
    {
        // shortcut co-location code if target already is a locality
        if (naming::is_locality(gid))
        {
            return hpx::post_p_cb<Action>(HPX_FORWARD(Continuation, cont), gid,
                actions::action_priority<Action>(), HPX_FORWARD(Callback, cb),
                HPX_FORWARD(Ts, vs)...);
        }

        // Attach the requested action as a continuation to a resolve_async
        // call on the locality responsible for the target gid.
        hpx::id_type service_target(
            agas::primary_namespace::get_service_instance(gid.get_gid()),
            hpx::id_type::management_type::unmanaged);

        using action_type = agas::server::primary_namespace::colocate_action;

        using placeholders::_2;
        return post_continue_cb<action_type>(
            util::functional::post_continuation(
                hpx::bind<Action>(
                    hpx::bind(util::functional::extract_locality(), _2, gid),
                    HPX_FORWARD(Ts, vs)...),
                HPX_FORWARD(Continuation, cont)),
            service_target, HPX_FORWARD(Callback, cb), gid.get_gid());
    }

    template <typename Continuation, typename Component, typename Signature,
        typename Derived, typename Callback, typename... Ts>
    bool post_colocated_cb(Continuation&& cont,
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        hpx::id_type const& gid, Callback&& cb, Ts&&... vs)
    {
        return post_colocated_cb<Derived>(HPX_FORWARD(Continuation, cont), gid,
            HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
    }
}    // namespace hpx::detail

#endif

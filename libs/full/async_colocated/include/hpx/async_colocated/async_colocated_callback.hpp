//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/agas_base/primary_namespace.hpp>
#include <hpx/agas_base/server/primary_namespace.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_colocated/async_colocated.hpp>
#include <hpx/async_colocated/async_colocated_callback_fwd.hpp>
#include <hpx/async_distributed/async_continue_callback.hpp>
#include <hpx/futures/traits/promise_local_result.hpp>

#include <utility>

namespace hpx { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback, typename... Ts>
    hpx::future<typename traits::promise_local_result<
        typename hpx::traits::extract_action<Action>::remote_result_type>::type>
    async_colocated_cb(hpx::id_type const& gid, Callback&& cb,
        Ts&&...
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        vs
#endif
    )
    {
#if defined(HPX_COMPUTE_DEVICE_CODE)
        HPX_UNUSED(gid);
        HPX_UNUSED(cb);
        HPX_ASSERT(false);
#else
        // Attach the requested action as a continuation to a resolve_async
        // call on the locality responsible for the target gid.
        hpx::id_type service_target(
            agas::primary_namespace::get_service_instance(gid.get_gid()),
            hpx::id_type::management_type::unmanaged);

        typedef typename hpx::traits::extract_action<Action>::remote_result_type
            remote_result_type;
        typedef agas::server::primary_namespace::colocate_action action_type;

        using placeholders::_2;
        return detail::async_continue_r_cb<action_type, remote_result_type>(
            util::functional::async_continuation(hpx::bind<Action>(
                hpx::bind(util::functional::extract_locality(), _2, gid),
                HPX_FORWARD(Ts, vs)...)),
            service_target, HPX_FORWARD(Callback, cb), gid.get_gid());
#endif
    }

    template <typename Component, typename Signature, typename Derived,
        typename Callback, typename... Ts>
    hpx::future<typename traits::promise_local_result<typename hpx::traits::
            extract_action<Derived>::remote_result_type>::type>
    async_colocated_cb(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        hpx::id_type const& gid, Callback&& cb, Ts&&... vs)
    {
        return async_colocated_cb<Derived>(
            gid, HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Continuation, typename Callback,
        typename... Ts>
    hpx::future<typename traits::promise_local_result<
        typename hpx::traits::extract_action<Action>::remote_result_type>::type>
    async_colocated_cb(
        Continuation&& cont, hpx::id_type const& gid, Callback&& cb,
        Ts&&...
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        vs
#endif
    )
    {
#if defined(HPX_COMPUTE_DEVICE_CODE)
        HPX_UNUSED(cont);
        HPX_UNUSED(gid);
        HPX_UNUSED(cb);
        HPX_ASSERT(false);
#else
        // Attach the requested action as a continuation to a resolve_async
        // call on the locality responsible for the target gid.
        hpx::id_type service_target(
            agas::primary_namespace::get_service_instance(gid.get_gid()),
            hpx::id_type::management_type::unmanaged);

        typedef typename hpx::traits::extract_action<Action>::remote_result_type
            remote_result_type;
        typedef agas::server::primary_namespace::colocate_action action_type;

        using placeholders::_2;
        return detail::async_continue_r_cb<action_type, remote_result_type>(
            util::functional::async_continuation(
                hpx::bind<Action>(
                    hpx::bind(util::functional::extract_locality(), _2, gid),
                    HPX_FORWARD(Ts, vs)...),
                HPX_FORWARD(Continuation, cont)),
            service_target, HPX_FORWARD(Callback, cb), gid.get_gid());
#endif
    }

    template <typename Continuation, typename Component, typename Signature,
        typename Derived, typename Callback, typename... Ts>
    hpx::future<typename traits::promise_local_result<typename hpx::traits::
            extract_action<Derived>::remote_result_type>::type>
    async_colocated_cb(Continuation&& cont,
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        hpx::id_type const& gid, Callback&& cb, Ts&&... vs)
    {
        return async_colocated_cb<Derived>(HPX_FORWARD(Continuation, cont), gid,
            HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
    }
}}    // namespace hpx::detail

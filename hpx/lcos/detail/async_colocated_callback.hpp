//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_ASYNC_COLOCATED_CALLBACK_MAR_30_2015_1146AM)
#define HPX_LCOS_ASYNC_COLOCATED_CALLBACK_MAR_30_2015_1146AM

#include <hpx/assertion.hpp>
#include <hpx/lcos/async_continue_callback.hpp>
#include <hpx/lcos/detail/async_colocated.hpp>
#include <hpx/lcos/detail/async_colocated_callback_fwd.hpp>
#include <hpx/runtime/agas/primary_namespace.hpp>
#include <hpx/runtime/agas/server/primary_namespace.hpp>
#include <hpx/traits/extract_action.hpp>
#include <hpx/traits/promise_local_result.hpp>

#include <utility>

namespace hpx { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::traits::extract_action<Action>::remote_result_type
        >::type>
    async_colocated_cb(naming::id_type const& gid, Callback&& cb, Ts&&... vs)
    {
#if defined(HPX_COMPUTE_DEVICE_CODE)
        HPX_ASSERT(false);
#else
        // Attach the requested action as a continuation to a resolve_async
        // call on the locality responsible for the target gid.
        naming::id_type service_target(
            agas::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);

        typedef
            typename hpx::traits::extract_action<Action>::remote_result_type
        remote_result_type;
        typedef agas::server::primary_namespace::colocate_action action_type;

        using util::placeholders::_2;
        return detail::async_continue_r_cb<action_type, remote_result_type>(
            util::functional::async_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<Ts>(vs)...)
                ),
            service_target, std::forward<Callback>(cb), gid.get_gid());
#endif
    }

    template <
        typename Component, typename Signature, typename Derived,
        typename Callback, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::traits::extract_action<Derived>::remote_result_type
        >::type>
    async_colocated_cb(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/
      , naming::id_type const& gid, Callback&& cb, Ts&&... vs)
    {
        return async_colocated_cb<Derived>(gid, std::forward<Callback>(cb),
            std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Continuation, typename Callback, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::traits::extract_action<Action>::remote_result_type
        >::type>
    async_colocated_cb(Continuation && cont,
        naming::id_type const& gid, Callback&& cb, Ts&&... vs)
    {
#if defined(HPX_COMPUTE_DEVICE_CODE)
        HPX_ASSERT(false);
#else
        // Attach the requested action as a continuation to a resolve_async
        // call on the locality responsible for the target gid.
        naming::id_type service_target(
            agas::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);

        typedef
            typename hpx::traits::extract_action<Action>::remote_result_type
        remote_result_type;
        typedef agas::server::primary_namespace::colocate_action action_type;

        using util::placeholders::_2;
        return detail::async_continue_r_cb<action_type, remote_result_type>(
            util::functional::async_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<Ts>(vs)...)
              , std::forward<Continuation>(cont)),
            service_target, std::forward<Callback>(cb), gid.get_gid());
#endif
    }

    template <
        typename Continuation,
        typename Component, typename Signature, typename Derived,
        typename Callback, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::traits::extract_action<Derived>::remote_result_type
        >::type>
    async_colocated_cb(
        Continuation && cont
      , hpx::actions::basic_action<Component, Signature, Derived> /*act*/
      , naming::id_type const& gid, Callback&& cb, Ts&&... vs)
    {
        return async_colocated_cb<Derived>(std::forward<Continuation>(cont), gid,
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }
}}

#endif

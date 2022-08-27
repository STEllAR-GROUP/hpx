//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/actions_base_support.hpp>
#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/actions_base/traits/is_continuation.hpp>
#include <hpx/agas_base/primary_namespace.hpp>
#include <hpx/agas_base/server/primary_namespace.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_colocated/async_colocated_fwd.hpp>
#include <hpx/async_colocated/functional/colocated_helpers.hpp>
#include <hpx/async_distributed/async_continue_fwd.hpp>
#include <hpx/async_distributed/bind_action.hpp>
#include <hpx/async_local/async_fwd.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/functional/move_only_function.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/promise_local_result.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/type_support/pack.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace detail {
    template <typename Action, typename Ts = typename Action::arguments_type>
    struct async_colocated_bound_action;

    template <typename Action, typename... Ts>
    struct async_colocated_bound_action<Action, hpx::tuple<Ts...>>
    {
        using type = hpx::detail::bound_action<Action,
            hpx::util::make_index_pack<1 + sizeof...(Ts)>,
            hpx::detail::bound<hpx::util::functional::extract_locality,
                hpx::util::index_pack<0, 1>, hpx::detail::placeholder<2ul>,
                hpx::id_type>,
            Ts...>;
    };
}}    // namespace hpx::detail

#define HPX_REGISTER_ASYNC_COLOCATED_DECLARATION(Action, Name)                 \
    HPX_UTIL_REGISTER_UNIQUE_FUNCTION_DECLARATION(                             \
        void(hpx::id_type, hpx::id_type),                                      \
        (hpx::util::functional::detail::async_continuation_impl<               \
            typename hpx::detail::async_colocated_bound_action<Action>::type,  \
            hpx::util::unused_type>),                                          \
        Name)                                                                  \
    /**/

#define HPX_REGISTER_ASYNC_COLOCATED(Action, Name)                             \
    HPX_UTIL_REGISTER_UNIQUE_FUNCTION(void(hpx::id_type, hpx::id_type),        \
        (hpx::util::functional::detail::async_continuation_impl<               \
            typename hpx::detail::async_colocated_bound_action<Action>::type,  \
            hpx::util::unused_type>),                                          \
        Name)                                                                  \
    /**/

namespace hpx { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename... Ts>
    hpx::future<typename traits::promise_local_result<
        typename hpx::traits::extract_action<Action>::remote_result_type>::type>
    async_colocated(hpx::id_type const& gid,
        Ts&&...
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        vs
#endif
    )
    {
        // Attach the requested action as a continuation to a resolve_async
        // call on the locality responsible for the target gid.
        hpx::id_type service_target(
            agas::primary_namespace::get_service_instance(gid.get_gid()),
            hpx::id_type::management_type::unmanaged);

#if !defined(HPX_COMPUTE_DEVICE_CODE)
        using remote_result_type =
            typename hpx::traits::extract_action<Action>::remote_result_type;
        using action_type = agas::server::primary_namespace::colocate_action;

        using placeholders::_2;
        return detail::async_continue_r<action_type, remote_result_type>(
            util::functional::async_continuation(hpx::bind<Action>(
                hpx::bind(util::functional::extract_locality(), _2, gid),
                HPX_FORWARD(Ts, vs)...)),
            service_target, gid.get_gid());
#else
        HPX_ASSERT(false);
        return hpx::future<typename traits::promise_local_result<typename hpx::
                traits::extract_action<Action>::remote_result_type>::type>{};
#endif
    }

    template <typename Component, typename Signature, typename Derived,
        typename... Ts>
    hpx::future<typename traits::promise_local_result<typename hpx::traits::
            extract_action<Derived>::remote_result_type>::type>
    async_colocated(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        hpx::id_type const& gid, Ts&&... vs)
    {
        return async_colocated<Derived>(gid, HPX_FORWARD(Ts, vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Continuation, typename... Ts>
    std::enable_if_t<traits::is_continuation_v<Continuation>,
        hpx::future<traits::promise_local_result_t<
            typename hpx::traits::extract_action<Action>::remote_result_type>>>
    async_colocated(Continuation&& cont, hpx::id_type const& gid,
        Ts&&...
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        vs
#endif
    )
    {
#if defined(HPX_COMPUTE_DEVICE_CODE)
        HPX_UNUSED(cont);
        HPX_UNUSED(gid);
        HPX_ASSERT(false);
#else
        // Attach the requested action as a continuation to a resolve_async
        // call on the locality responsible for the target gid.
        hpx::id_type service_target(
            agas::primary_namespace::get_service_instance(gid.get_gid()),
            hpx::id_type::management_type::unmanaged);

        using remote_result_type =
            typename hpx::traits::extract_action<Action>::remote_result_type;
        using action_type = agas::server::primary_namespace::colocate_action;

        using placeholders::_2;
        return detail::async_continue_r<action_type, remote_result_type>(
            util::functional::async_continuation(
                hpx::bind<Action>(
                    hpx::bind(util::functional::extract_locality(), _2, gid),
                    HPX_FORWARD(Ts, vs)...),
                HPX_FORWARD(Continuation, cont)),
            service_target, gid.get_gid());
#endif
    }

    template <typename Continuation, typename Component, typename Signature,
        typename Derived, typename... Ts>
    std::enable_if_t<traits::is_continuation_v<Continuation>,
        hpx::future<traits::promise_local_result_t<
            typename hpx::traits::extract_action<Derived>::remote_result_type>>>
    async_colocated(Continuation&& cont,
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        hpx::id_type const& gid, Ts&&... vs)
    {
        return async_colocated<Derived>(
            HPX_FORWARD(Continuation, cont), gid, HPX_FORWARD(Ts, vs)...);
    }
}}    // namespace hpx::detail

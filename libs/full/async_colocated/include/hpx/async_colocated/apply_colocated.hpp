//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/actions_base/actions_base_support.hpp>
#include <hpx/agas/primary_namespace.hpp>
#include <hpx/agas/server/primary_namespace.hpp>
#include <hpx/async_colocated/apply_colocated_fwd.hpp>
#include <hpx/async_distributed/applier/apply_continue.hpp>
#include <hpx/async_distributed/applier/register_apply_colocated.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/traits/is_continuation.hpp>
#include <hpx/util/bind_action.hpp>
#include <hpx/util/functional/colocated_helpers.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename... Ts>
    bool apply_colocated(naming::id_type const& gid, Ts&&... vs)
    {
        // shortcut co-location code if target already is a locality
        if (naming::is_locality(gid))
        {
            return apply<Action>(gid, std::forward<Ts>(vs)...);
        }

        // Attach the requested action as a continuation to a resolve_async
        // call on the locality responsible for the target gid.
        naming::id_type service_target(
            agas::primary_namespace::get_service_instance(gid.get_gid()),
            naming::id_type::unmanaged);

        typedef agas::server::primary_namespace::colocate_action action_type;

        using util::placeholders::_2;
        return apply_continue<action_type>(
            util::functional::apply_continuation(util::bind<Action>(
                util::bind(util::functional::extract_locality(), _2, gid),
                std::forward<Ts>(vs)...)),
            service_target, gid.get_gid());
    }

    template <typename Component, typename Signature, typename Derived,
        typename... Ts>
    bool apply_colocated(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        naming::id_type const& gid, Ts&&... vs)
    {
        return apply_colocated<Derived>(gid, std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Continuation, typename... Ts>
    typename std::enable_if<traits::is_continuation<Continuation>::value,
        bool>::type
    apply_colocated(Continuation&& cont, naming::id_type const& gid, Ts&&... vs)
    {
        // shortcut co-location code if target already is a locality
        if (naming::is_locality(gid))
        {
            return apply_continue<Action>(
                std::forward<Continuation>(cont), gid, std::forward<Ts>(vs)...);
        }

        // Attach the requested action as a continuation to a resolve_async
        // call on the locality responsible for the target gid.
        naming::id_type service_target(
            agas::primary_namespace::get_service_instance(gid.get_gid()),
            naming::id_type::unmanaged);

        typedef agas::server::primary_namespace::colocate_action action_type;

        using util::placeholders::_2;
        return apply_continue<action_type>(
            util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid),
                    std::forward<Ts>(vs)...),
                std::forward<Continuation>(cont)),
            service_target, gid.get_gid());
    }

    template <typename Continuation, typename Component, typename Signature,
        typename Derived, typename... Ts>
    bool apply_colocated(Continuation&& cont,
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        naming::id_type const& gid, Ts&&... vs)
    {
        return apply_colocated<Derived>(
            std::forward<Continuation>(cont), gid, std::forward<Ts>(vs)...);
    }
}}    // namespace hpx::detail

#endif

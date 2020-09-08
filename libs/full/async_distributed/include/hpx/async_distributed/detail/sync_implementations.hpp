//  Copyright (c) 2007-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/traits/action_select_direct_execution.hpp>
#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_distributed/detail/async_implementations.hpp>
#include <hpx/async_distributed/detail/sync_implementations_fwd.hpp>
#include <hpx/async_local/sync_fwd.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/traits/action_was_object_migrated.hpp>
#include <hpx/traits/component_supports_migration.hpp>

#include <cstddef>
#include <utility>

namespace hpx { namespace detail {
    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Result>
    struct sync_local_invoke_direct
    {
        template <typename... Ts>
        HPX_FORCEINLINE static Result call(
            naming::id_type const& /*id*/, naming::address&& addr, Ts&&... vs)
        {
            typedef typename Action::remote_result_type remote_result_type;

            typedef traits::get_remote_result<Result, remote_result_type>
                get_remote_result_type;

            return get_remote_result_type::call(Action::execute_function(
                addr.address_, addr.type_, std::forward<Ts>(vs)...));
        }
    };

    template <typename Action>
    struct sync_local_invoke_direct<Action, void>
    {
        template <typename... Ts>
        HPX_FORCEINLINE static void call(
            naming::id_type const& /*id*/, naming::address&& addr, Ts&&... vs)
        {
            Action::execute_function(
                addr.address_, addr.type_, std::forward<Ts>(vs)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Launch, typename... Ts>
    typename hpx::traits::extract_action<Action>::type::local_result_type
    sync_impl(Launch&& policy, hpx::id_type const& id, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        typedef typename action_type::local_result_type result_type;
        typedef typename action_type::component_type component_type;

        std::pair<bool, components::pinned_ptr> r;

        naming::address addr;
        if (agas::is_local_address_cached(id, addr) &&
            can_invoke_locally<action_type>())
        {
            // route launch policy through component
            launch adapted_policy =
                traits::action_select_direct_execution<Action>::call(
                    policy, addr.address_);

            if (traits::component_supports_migration<component_type>::call())
            {
                r = traits::action_was_object_migrated<Action>::call(
                    id, addr.address_);
                if (!r.first)
                {
                    if (adapted_policy == launch::sync ||
                        action_type::direct_execution::value)
                    {
                        return hpx::detail::sync_local_invoke_direct<
                            action_type, result_type>::call(id, std::move(addr),
                            std::forward<Ts>(vs)...);
                    }
                }
            }
            else if (adapted_policy == launch::sync ||
                action_type::direct_execution::value)
            {
                return hpx::detail::sync_local_invoke_direct<action_type,
                    result_type>::call(id, std::move(addr),
                    std::forward<Ts>(vs)...);
            }
        }

        return async_remote_impl<Action>(std::forward<Launch>(policy), id,
            std::move(addr), std::forward<Ts>(vs)...)
            .get();
    }
    /// \endcond
}}    // namespace hpx::detail

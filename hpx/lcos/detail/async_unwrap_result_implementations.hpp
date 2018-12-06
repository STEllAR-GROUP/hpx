//  Copyright (c) 2007-2018 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_ASYNC_UNWRAP_IMPLEMENTATIONS_JUL_22_2018_0137PM)
#define HPX_LCOS_ASYNC_UNWRAP_IMPLEMENTATIONS_JUL_22_2018_0137PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/detail/async_implementations.hpp>
#include <hpx/lcos/detail/async_unwrap_result_implementations_fwd.hpp>
#include <hpx/lcos/detail/sync_implementations.hpp>
#include <hpx/lcos/packaged_action.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/traits/action_decorate_function.hpp>
#include <hpx/traits/action_was_object_migrated.hpp>
#include <hpx/traits/action_select_direct_execution.hpp>
#include <hpx/traits/component_supports_migration.hpp>
#include <hpx/traits/component_type_is_compatible.hpp>
#include <hpx/traits/extract_action.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/util/assert.hpp>

#include <cstddef>
#include <utility>

namespace hpx { namespace detail
{
    /// \cond NOINTERNAL
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename... Ts>
    typename hpx::traits::extract_action<Action>::type::local_result_type
    async_local_unwrap_impl(launch policy, hpx::id_type const& id,
        naming::address& addr, std::pair<bool, components::pinned_ptr>& r,
        Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        typedef typename action_type::local_result_type result_type;

        if (policy == launch::sync || action_type::direct_execution::value)
        {
            return hpx::detail::sync_local_invoke_direct<
                        action_type, result_type
                    >::call(id, std::move(addr), std::forward<Ts>(vs)...);
        }
        else if (hpx::detail::has_async_policy(policy))
        {
            return keep_alive(
                hpx::async(action_invoker<action_type>(), addr.address_,
                    addr.type_, std::forward<Ts>(vs)...),
                id, std::move(r.second));
        }

        HPX_ASSERT(policy == launch::deferred);

        return keep_alive(
            hpx::async(launch::deferred, action_invoker<action_type>(),
                addr.address_, addr.type_, std::forward<Ts>(vs)...),
            id, std::move(r.second));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Result, typename... Ts>
    bool async_local_unwrap_impl_all(launch policy, hpx::id_type const& id,
        naming::address& addr, std::pair<bool, components::pinned_ptr>& r,
        Result& result, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        //typedef typename action_type::local_result_type result_type;
        typedef typename action_type::component_type component_type;

        // route launch policy through component
        policy = traits::action_select_direct_execution<Action>::call(
            policy, addr.address_);

        if (traits::component_supports_migration<component_type>::call())
        {
            r = traits::action_was_object_migrated<Action>::call(
                    id, addr.address_);

            if (!r.first)
            {
                result = async_local_unwrap_impl<Action>(
                    policy, id, addr, r, std::forward<Ts>(vs)...);

                return true;
            }

            // can't locally handle object if it is currently being migrated
            return false;
        }

        result = async_local_unwrap_impl<Action>(
            policy, id, addr, r, std::forward<Ts>(vs)...);

        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Launch, typename ...Ts>
    typename hpx::traits::extract_action<Action>::type::local_result_type
    async_unwrap_result_impl(Launch && policy, hpx::id_type const& id, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        typedef typename action_type::local_result_type result_type;
        //typedef typename action_type::component_type component_type;

        std::pair<bool, components::pinned_ptr> r;

        naming::address addr;
        if (agas::is_local_address_cached(id, addr) &&
            can_invoke_locally<action_type>())
        {
            result_type result;
            if (async_local_unwrap_impl_all<Action>(
                    policy, id, addr, r, result, std::forward<Ts>(vs)...))
            {
                return result;
            }
        }

        // the asynchronous result is auto-unwrapped by the return type
        return async_remote_impl<Action>(std::forward<Launch>(policy), id,
            std::move(addr), std::forward<Ts>(vs)...);
    }
    /// \endcond
}}

#endif

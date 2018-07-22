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
    template <typename Action, typename ...Ts>
    typename hpx::traits::extract_action<Action>::type::local_result_type
    async_unwrap_result_impl(launch policy, hpx::id_type const& id, Ts&&... vs)
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
            policy = traits::action_select_direct_execution<Action>::call(
                policy, addr.address_);

            if (traits::component_supports_migration<component_type>::call())
            {
                r = traits::action_was_object_migrated<Action>::call(
                        id, addr.address_);
                if (!r.first)
                {
                    if (policy == launch::sync ||
                        action_type::direct_execution::value)
                    {
                        return hpx::detail::sync_local_invoke_direct<
                                action_type, result_type
                        >::call(id, std::move(addr), std::forward<Ts>(vs)...);
                    }
                }
            }
            else if (policy == launch::sync ||
                action_type::direct_execution::value)
            {
                return hpx::detail::sync_local_invoke_direct<
                        action_type, result_type
                    >::call(id, std::move(addr), std::forward<Ts>(vs)...);
            }
        }

        // the asynchronous result is auto-unwrapped by the return type
        future<result_type> f;
        {
            handle_managed_target<result_type> hmt(id, f);

            if (policy == launch::sync ||
                hpx::detail::has_async_policy(policy))
            {
                lcos::packaged_action<action_type, result_type> p;

                f = p.get_future();
                p.apply(std::move(addr), hmt.get_id(), std::forward<Ts>(vs)...);
                if (policy == launch::sync)
                    f.wait();
            }
            else if (policy == launch::deferred)
            {
                lcos::packaged_action<action_type, result_type> p;

                f = p.get_future();
                p.apply_deferred(std::move(addr), hmt.get_id(),
                    std::forward<Ts>(vs)...);
            }
            else
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "async_impl", "unknown launch policy");
                return f;
            }
        }
        return f;
    }

    template <typename Action, typename ...Ts>
    typename hpx::traits::extract_action<Action>::type::local_result_type
    async_unwrap_result_impl(hpx::detail::sync_policy, hpx::id_type const& id,
        Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        typedef typename action_type::local_result_type result_type;
        typedef typename action_type::component_type component_type;

        std::pair<bool, components::pinned_ptr> r;

        naming::address addr;
        if (agas::is_local_address_cached(id, addr) &&
            can_invoke_locally<action_type>())
        {
            if (traits::component_supports_migration<component_type>::call())
            {
                r = traits::action_was_object_migrated<Action>::call(
                        id, addr.address_);
                if (!r.first)
                {
                    return hpx::detail::sync_local_invoke_direct<
                            action_type, result_type
                        >::call(id, std::move(addr), std::forward<Ts>(vs)...);
                }
            }
            else
            {
                return hpx::detail::sync_local_invoke_direct<
                        action_type, result_type
                    >::call(id, std::move(addr), std::forward<Ts>(vs)...);
            }
        }

        // the asynchronous result is auto-unwrapped by the return type
        future<result_type> f;
        {
            handle_managed_target<result_type> hmt(id, f);
            lcos::packaged_action<action_type, result_type> p;

            f = p.get_future();
            p.apply(std::move(addr), hmt.get_id(), std::forward<Ts>(vs)...);
            f.wait();
        }
        return f;
    }

    template <typename Action, typename... Ts>
    typename hpx::traits::extract_action<Action>::type::local_result_type
    async_unwrap_result_impl(hpx::detail::async_policy async_policy,
        hpx::id_type const& id, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        typedef typename action_type::local_result_type result_type;
        typedef typename action_type::component_type component_type;

        future<result_type> f;
        std::pair<bool, components::pinned_ptr> r;

        naming::address addr;
        if (agas::is_local_address_cached(id, addr) &&
            can_invoke_locally<action_type>())
        {
            // route launch policy through component
            launch policy = traits::action_select_direct_execution<Action>::call(
                async_policy, addr.address_);

            if (traits::component_supports_migration<component_type>::call())
            {
                r = traits::action_was_object_migrated<Action>::call(
                        id, addr.address_);
                if (!r.first)
                {
                    if (policy == launch::sync ||
                        action_type::direct_execution::value)
                    {
                        return sync_local_invoke_direct<
                                action_type, result_type
                            >::call(id, std::move(addr), std::forward<Ts>(vs)...);
                    }
                    else
                    {
                        f = hpx::async(action_invoker<action_type>(),
                                addr.address_, addr.type_, std::forward<Ts>(vs)...);

                        return keep_alive(std::move(f), id, std::move(r.second));
                    }
                }
            }
            else
            {
                if (policy == launch::sync ||
                    action_type::direct_execution::value)
                {
                    return sync_local_invoke_direct<
                            action_type, result_type
                        >::call(id, std::move(addr), std::forward<Ts>(vs)...);
                }
                else
                {
                    f = hpx::async(action_invoker<action_type>(), addr.address_,
                            addr.type_, std::forward<Ts>(vs)...);
                    return keep_alive(std::move(f), id);
                }
            }
        }

        {
            handle_managed_target<result_type> hmt(id, f);
            lcos::packaged_action<action_type, result_type> p;

            f = p.get_future();
            p.apply(std::move(addr), hmt.get_id(), std::forward<Ts>(vs)...);
        }

        return f;
    }

    template <typename Action, typename ...Ts>
    typename hpx::traits::extract_action<Action>::type::local_result_type
    async_unwrap_result_impl(hpx::launch::deferred_policy,
        hpx::id_type const& id, Ts&&... vs)
    {
        return async_impl(hpx::launch::deferred, id, std::forward<Ts>(vs)...);
    }
    /// \endcond
}}

#endif

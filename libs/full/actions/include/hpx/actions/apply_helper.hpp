//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions/apply_helper_fwd.hpp>
#include <hpx/actions/traits/action_continuation.hpp>
#include <hpx/actions/traits/action_decorate_continuation.hpp>
#include <hpx/actions_base/actions_base_support.hpp>
#include <hpx/actions_base/traits/action_priority.hpp>
#include <hpx/actions_base/traits/action_schedule_thread.hpp>
#include <hpx/actions_base/traits/action_select_direct_execution.hpp>
#include <hpx/actions_base/traits/action_stacksize.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/state.hpp>
#include <hpx/threading_base/thread_helpers.hpp>

#include <chrono>
#include <exception>
#include <memory>
#include <thread>
#include <utility>

namespace hpx {
    bool HPX_EXPORT is_pre_startup();
}

namespace hpx { namespace applier { namespace detail {

    ///////////////////////////////////////////////////////////////////////
    template <typename Action>
    inline threads::thread_priority fix_priority(
        threads::thread_priority priority)
    {
        return hpx::actions::detail::thread_priority<
            static_cast<threads::thread_priority>(
                traits::action_priority<Action>::value)>::call(priority);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename... Ts>
    void call_async(threads::thread_init_data&& data,
        naming::id_type const& target, naming::address::address_type lva,
        naming::address::component_type comptype,
        threads::thread_priority priority, Ts&&... vs)
    {
        using continuation_type =
            typename traits::action_continuation<Action>::type;

        continuation_type cont;
        if (traits::action_decorate_continuation<Action>::call(cont))    //-V614
        {
            data.func = Action::construct_thread_function(target,
                std::move(cont), lva, comptype, std::forward<Ts>(vs)...);
        }
        else
        {
            data.func = Action::construct_thread_function(
                target, lva, comptype, std::forward<Ts>(vs)...);
        }

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
        data.description =
            util::thread_description(actions::detail::get_action_name<Action>(),
                actions::detail::get_action_name_itt<Action>());
#else
        data.description = actions::detail::get_action_name<Action>();
#endif
#endif
        data.priority = fix_priority<Action>(priority);
        data.stacksize = traits::action_stacksize<Action>::value;

        while (!threads::threadmanager_is_at_least(state_running))
        {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(HPX_NETWORK_RETRIES_SLEEP));
        }

        traits::action_schedule_thread<Action>::call(lva, comptype, data);
    }

    template <typename Action, typename Continuation, typename... Ts>
    void call_async(threads::thread_init_data&& data, Continuation&& cont,
        naming::id_type const& target, naming::address::address_type lva,
        naming::address::component_type comptype,
        threads::thread_priority priority, Ts&&... vs)
    {
        // first decorate the continuation
        traits::action_decorate_continuation<Action>::call(cont);

        // now, schedule the thread
        data.func = Action::construct_thread_function(target,
            std::forward<Continuation>(cont), lva, comptype,
            std::forward<Ts>(vs)...);

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
        data.description =
            util::thread_description(actions::detail::get_action_name<Action>(),
                actions::detail::get_action_name_itt<Action>());
#else
        data.description = actions::detail::get_action_name<Action>();
#endif
#endif
        data.priority = fix_priority<Action>(priority);
        data.stacksize = static_cast<threads::thread_stacksize>(
            traits::action_stacksize<Action>::value);

        while (!threads::threadmanager_is_at_least(state_running))
        {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(HPX_NETWORK_RETRIES_SLEEP));
        }

        traits::action_schedule_thread<Action>::call(lva, comptype, data);
    }

    template <typename Action, typename... Ts>
    HPX_FORCEINLINE void call_sync(naming::address::address_type lva,
        naming::address::component_type comptype, Ts&&... vs)
    {
        Action::execute_function(lva, comptype, std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Continuation, typename... Ts>
    HPX_FORCEINLINE void call_sync(Continuation&& cont,
        naming::address::address_type lva,
        naming::address::component_type comptype, Ts&&... vs)
    {
        try
        {
            cont.trigger_value(Action::execute_function(
                lva, comptype, std::forward<Ts>(vs)...));
        }
        catch (...)
        {
            // make sure hpx::exceptions are propagated back to the
            // client
            cont.trigger_error(std::current_exception());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct apply_helper<Action, /*DirectExecute=*/false>
    {
        template <typename... Ts>
        static void call(threads::thread_init_data&& data,
            naming::id_type const& target, naming::address::address_type lva,
            naming::address::component_type comptype,
            threads::thread_priority priority, Ts&&... vs)
        {
            // route launch policy through component
            launch policy =
                traits::action_select_direct_execution<Action>::call(
                    launch::async, lva);

            if (policy == launch::async)
            {
                call_async<Action>(std::move(data), target, lva, comptype,
                    priority, std::forward<Ts>(vs)...);
            }
            else
            {
                call_sync<Action>(lva, comptype, std::forward<Ts>(vs)...);
            }
        }

        template <typename Continuation, typename... Ts>
        static void call(threads::thread_init_data&& data, Continuation&& cont,
            naming::id_type const& target, naming::address::address_type lva,
            naming::address::component_type comptype,
            threads::thread_priority priority, Ts&&... vs)
        {
            // route launch policy through component
            launch policy =
                traits::action_select_direct_execution<Action>::call(
                    launch::async, lva);

            if (policy == launch::async)
            {
                call_async<Action>(std::move(data),
                    std::forward<Continuation>(cont), target, lva, comptype,
                    priority, std::forward<Ts>(vs)...);
            }
            else
            {
                call_sync<Action>(std::forward<Continuation>(cont), lva,
                    comptype, std::forward<Ts>(vs)...);
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct apply_helper<Action, /*DirectExecute=*/true>
    {
        // If local and to be directly executed, just call the function
        template <typename... Ts>
        HPX_FORCEINLINE static void call(threads::thread_init_data&& data,
            naming::id_type const& target, naming::address::address_type lva,
            naming::address::component_type comptype,
            threads::thread_priority priority, Ts&&... vs)
        {
            // Direct actions should be able to be executed from a
            // non-HPX thread as well
            if (this_thread::has_sufficient_stack_space() ||
                !threads::threadmanager_is_at_least(state_running))
            {
                call_sync<Action>(lva, comptype, std::forward<Ts>(vs)...);
            }
            else
            {
                call_async<Action>(std::move(data), target, lva, comptype,
                    priority, std::forward<Ts>(vs)...);
            }
        }

        template <typename Continuation, typename... Ts>
        HPX_FORCEINLINE static void call(threads::thread_init_data&& data,
            Continuation&& cont, naming::id_type const& target,
            naming::address::address_type lva,
            naming::address::component_type comptype,
            threads::thread_priority priority, Ts&&... vs)
        {
            // Direct actions should be able to be executed from a
            // non-HPX thread as well
            if (this_thread::has_sufficient_stack_space() ||
                !threads::threadmanager_is_at_least(state_running))
            {
                call_sync<Action>(std::forward<Continuation>(cont), lva,
                    comptype, std::forward<Ts>(vs)...);
            }
            else
            {
                call_async<Action>(std::move(data),
                    std::forward<Continuation>(cont), target, lva, comptype,
                    priority, std::forward<Ts>(vs)...);
            }
        }
    };
}}}    // namespace hpx::applier::detail

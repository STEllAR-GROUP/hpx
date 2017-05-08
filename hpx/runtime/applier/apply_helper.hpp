//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLIER_APPLY_HELPER_JUN_25_2008_0917PM)
#define HPX_APPLIER_APPLY_HELPER_JUN_25_2008_0917PM

#include <hpx/config.hpp>
#include <hpx/compat/thread.hpp>
#include <hpx/state.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/traits/action_decorate_continuation.hpp>
#include <hpx/traits/action_continuation.hpp>
#include <hpx/traits/action_priority.hpp>
#include <hpx/traits/action_schedule_thread.hpp>
#include <hpx/traits/action_stacksize.hpp>
#include <hpx/util/decay.hpp>

#include <chrono>
#include <memory>
#include <utility>

namespace hpx {
    bool HPX_EXPORT is_pre_startup();
}

namespace hpx { namespace applier { namespace detail
{
    ///////////////////////////////////////////////////////////////////////
    template <typename Action>
    inline threads::thread_priority
        fix_priority(threads::thread_priority priority)
    {
        return hpx::actions::detail::thread_priority<
            static_cast<threads::thread_priority>(
                traits::action_priority<Action>::value)
        >::call(priority);
    }

    ///////////////////////////////////////////////////////////////////////
    template <typename Action,
        bool DirectExecute = Action::direct_execution::value>
    struct apply_helper;

    template <typename Action>
    struct apply_helper<Action, /*DirectExecute=*/false>
    {
        template <typename ...Ts>
        static void
        call (threads::thread_init_data&& data, naming::id_type const& target,
            naming::address::address_type lva,
            threads::thread_priority priority, Ts&&... vs)
        {
            typedef typename traits::action_continuation<Action>::type
                continuation_type;

            continuation_type cont;
            if (traits::action_decorate_continuation<Action>::call(cont)) //-V614
            {
                data.func = Action::construct_thread_function(target, std::move(cont),
                    lva, std::forward<Ts>(vs)...);
            }
            else
            {
                data.func = Action::construct_thread_function(target, lva,
                    std::forward<Ts>(vs)...);
            }

#if defined(HPX_HAVE_THREAD_TARGET_ADDRESS)
            data.lva = lva;
#endif
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
            data.description = util::thread_description(
                actions::detail::get_action_name<Action>(),
                actions::detail::get_action_name_itt<Action>());
#else
            data.description = actions::detail::get_action_name<Action>();
#endif
#endif
            data.priority = fix_priority<Action>(priority);
            data.stacksize = threads::get_stack_size(
                static_cast<threads::thread_stacksize>(
                    traits::action_stacksize<Action>::value));

            while (!threads::threadmanager_is_at_least(state_running))
            {
                compat::this_thread::sleep_for(
                    std::chrono::milliseconds(HPX_NETWORK_RETRIES_SLEEP));
            }

            traits::action_schedule_thread<Action>::call(
                lva, data, threads::pending);
        }

        template <typename Continuation, typename ...Ts>
        static void
        call (threads::thread_init_data&& data, Continuation && cont,
            naming::id_type const& target, naming::address::address_type lva,
            threads::thread_priority priority, Ts&&... vs)
        {
            // first decorate the continuation
            traits::action_decorate_continuation<Action>::call(cont);

            // now, schedule the thread
            data.func = Action::construct_thread_function(target, std::move(cont), lva,
                std::forward<Ts>(vs)...);
#if defined(HPX_HAVE_THREAD_TARGET_ADDRESS)
            data.lva = lva;
#endif
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
            data.description = util::thread_description(
                actions::detail::get_action_name<Action>(),
                actions::detail::get_action_name_itt<Action>());
#else
            data.description = actions::detail::get_action_name<Action>();
#endif
#endif
            data.priority = fix_priority<Action>(priority);
            data.stacksize = threads::get_stack_size(
                static_cast<threads::thread_stacksize>(
                    traits::action_stacksize<Action>::value));

            while (!threads::threadmanager_is_at_least(state_running))
            {
                compat::this_thread::sleep_for(
                    std::chrono::milliseconds(HPX_NETWORK_RETRIES_SLEEP));
            }

            traits::action_schedule_thread<Action>::call(
                lva, data, threads::pending);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct apply_helper<Action, /*DirectExecute=*/true>
    {
        // If local and to be directly executed, just call the function
        template <typename ...Ts>
        HPX_FORCEINLINE static void
        call (threads::thread_init_data&& data, naming::id_type const& target,
            naming::address::address_type lva,
            threads::thread_priority priority, Ts &&... vs)
        {
            // Direct actions should be able to be executed from a non-HPX thread
            // as well
            if (this_thread::has_sufficient_stack_space() ||
                !threads::threadmanager_is_at_least(state_running))
            {
                Action::execute_function(lva, std::forward<Ts>(vs)...);
            }
            else
            {
                apply_helper<Action, false>::call(std::move(data), target, lva,
                    priority, std::forward<Ts>(vs)...);
            }
        }

        template <typename Continuation, typename ...Ts>
        HPX_FORCEINLINE static void
        call (threads::thread_init_data&& data, Continuation && cont,
            naming::id_type const& target, naming::address::address_type lva,
            threads::thread_priority priority, Ts &&... vs)
        {
            // Direct actions should be able to be executed from a non-HPX thread
            // as well
            if (this_thread::has_sufficient_stack_space() ||
                !threads::threadmanager_is_at_least(state_running))
            {
                try {
                    cont.trigger_value(Action::execute_function(lva,
                        std::forward<Ts>(vs)...));
                }
                catch (...) {
                    // make sure hpx::exceptions are propagated back to the
                    // client
                    cont.trigger_error(boost::current_exception());
                }
            }
            else
            {
                apply_helper<Action, false>::call(std::move(data),
                    std::forward<Continuation>(cont),
                    target, lva, priority, std::forward<Ts>(vs)...);
            }
        }
    };
}}}

#endif

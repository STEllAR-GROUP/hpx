//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLIER_APPLY_HELPER_JUN_25_2008_0917PM)
#define HPX_APPLIER_APPLY_HELPER_JUN_25_2008_0917PM

#include <hpx/config.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/traits/action_decorate_continuation.hpp>
#include <hpx/traits/action_priority.hpp>
#include <hpx/traits/action_schedule_thread.hpp>
#include <hpx/traits/action_stacksize.hpp>
#include <hpx/traits/action_direct_execution.hpp>
#include <hpx/util/decay.hpp>

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
        bool DirectExecute = traits::action_direct_execution<Action>::value>
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
            std::unique_ptr<actions::continuation> cont;
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
            data.description = actions::detail::get_action_name<Action>();
#endif
            data.priority = fix_priority<Action>(priority);
            data.stacksize = threads::get_stack_size(
                static_cast<threads::thread_stacksize>(
                    traits::action_stacksize<Action>::value));

            traits::action_schedule_thread<Action>::call(
                lva, data, threads::pending);
        }

        template <typename Continuation, typename ...Ts>
        static void
        call (threads::thread_init_data&& data, Continuation && cont,
            naming::id_type const& target, naming::address::address_type lva,
            threads::thread_priority priority, Ts&&... vs)
        {
            std::unique_ptr<actions::continuation> c(
                new typename util::decay<Continuation>::type(
                    std::forward<Continuation>(cont)));
            call(std::move(data), std::move(c), target, lva, priority,
                std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        static void
        call (threads::thread_init_data&& data,
            std::unique_ptr<actions::continuation> cont,
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
            data.description = actions::detail::get_action_name<Action>();
#endif
            data.priority = fix_priority<Action>(priority);
            data.stacksize = threads::get_stack_size(
                static_cast<threads::thread_stacksize>(
                    traits::action_stacksize<Action>::value));

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
                hpx::threads::get_self_ptr() == nullptr)
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
        call (threads::thread_init_data&& data, Continuation && c,
            naming::id_type const& target, naming::address::address_type lva,
            threads::thread_priority priority, Ts &&... vs)
        {
            std::unique_ptr<actions::continuation> cont(
                new typename util::decay<Continuation>::type(
                    std::forward<Continuation>(c)));

            call(std::move(data), std::move(cont), target, lva, priority,
                std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        HPX_FORCEINLINE static void
        call (threads::thread_init_data&& data,
            std::unique_ptr<actions::continuation> cont,
            naming::id_type const& target, naming::address::address_type lva,
            threads::thread_priority priority, Ts &&... vs)
        {
            // Direct actions should be able to be executed from a non-HPX thread
            // as well
            if (this_thread::has_sufficient_stack_space() ||
                hpx::threads::get_self_ptr() == nullptr)
            {
                try {
                    cont->trigger(Action::execute_function(lva,
                        std::forward<Ts>(vs)...));
                }
                catch (...) {
                    // make sure hpx::exceptions are propagated back to the
                    // client
                    cont->trigger_error(boost::current_exception());
                }
            }
            else
            {
                apply_helper<Action, false>::call(std::move(data), std::move(cont),
                    target, lva, priority, std::forward<Ts>(vs)...);
            }
        }
    };
}}}

#endif

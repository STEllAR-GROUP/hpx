//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/components_base/get_lva.hpp>
#include <hpx/components_base/traits/action_decorate_function.hpp>
#include <hpx/coroutines/coroutine.hpp>
#include <hpx/execution_base/register_locks.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/thread_support/unlock_guard.hpp>
#include <hpx/threading_base/thread_data.hpp>

#include <mutex>
#include <type_traits>
#include <utility>

namespace hpx { namespace components
{
    /// This hook can be inserted into the derivation chain of any component
    /// allowing to automatically lock all action invocations for any instance
    /// of the given component.
    template <typename BaseComponent, typename Mutex = lcos::local::spinlock>
    struct locking_hook : BaseComponent
    {
    private:
        typedef Mutex mutex_type;
        typedef BaseComponent base_type;
        typedef typename base_type::this_component_type this_component_type;

    public:
        template <typename ...Arg>
        locking_hook(Arg &&... arg)
          : base_type(std::forward<Arg>(arg)...)
        {}

        locking_hook(locking_hook const& rhs)
          : base_type(rhs)
          , mtx_()
        {}
        locking_hook(locking_hook && rhs)
          : base_type(std::move(rhs))
          , mtx_()
        {}

        using decorates_action = void;

        locking_hook& operator=(locking_hook const& rhs)
        {
            this->base_type::operator=(rhs);
            return *this;
        }
        locking_hook& operator=(locking_hook&& rhs)
        {
            this->base_type::operator=(std::move(rhs));
            return *this;
        }

        /// This is the hook implementation for decorate_action which locks
        /// the component ensuring that only one action is executed at a time
        /// for this component instance.
        template <typename F>
        static threads::thread_function_type
        decorate_action(naming::address_type lva, F && f)
        {
            return util::one_shot(util::bind_front(
                &locking_hook::thread_function,
                get_lva<this_component_type>::call(lva),
                traits::component_decorate_function<base_type>::call(
                    lva, std::forward<F>(f))));
        }

    protected:
        typedef util::function_nonser<
            threads::thread_arg_type(threads::thread_result_type)
        > yield_decorator_type;

        struct decorate_wrapper
        {
            template <typename F, typename Enable = typename
                std::enable_if<!std::is_same<typename std::decay<F>::type,
                    decorate_wrapper>::value>::type>
            // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
            decorate_wrapper(F && f)
            {
                threads::get_self().decorate_yield(std::forward<F>(f));
            }

            ~decorate_wrapper()
            {
                threads::get_self().undecorate_yield();
            }
        };

        // Execute the wrapped action. This locks the mutex ensuring a thread
        // safe action invocation.
        threads::thread_result_type thread_function(
            threads::thread_function_type f, threads::thread_arg_type state)
        {
            threads::thread_result_type result(
                threads::thread_schedule_state::unknown,
                threads::invalid_thread_id);

            // now lock the mutex and execute the action
            std::unique_lock<mutex_type> l(mtx_);

            // We can safely ignore this lock while checking as it is
            // guaranteed to be unlocked before the thread is suspended.
            //
            // If this lock is not ignored it will cause false positives as the
            // check for held locks is performed before this lock is unlocked.
            util::ignore_while_checking<
                    std::unique_lock<mutex_type>
                > ignore_lock(&l);

            {
                // register our yield decorator
                decorate_wrapper yield_decorator(
                    util::bind_front(&locking_hook::yield_function, this));

                result = f(state);

                (void)yield_decorator;       // silence gcc warnings
            }

            return result;
        }

        struct undecorate_wrapper
        {
            undecorate_wrapper()
              : yield_decorator_(threads::get_self().undecorate_yield())
            {}

            ~undecorate_wrapper()
            {
                threads::get_self().decorate_yield(std::move(yield_decorator_));
            }

            yield_decorator_type yield_decorator_;
        };

        // The yield decorator unlocks the mutex and calls the system yield
        // which gives up control back to the thread manager.
        threads::thread_arg_type yield_function(threads::thread_result_type state)
        {
            // We un-decorate the yield function as the lock handling may
            // suspend, which causes an infinite recursion otherwise.
            undecorate_wrapper yield_decorator;
            threads::thread_arg_type result =
                threads::thread_restart_state::unknown;

            {
                util::unlock_guard<mutex_type> ul(mtx_);
                result = threads::get_self().yield_impl(state);
            }

            // Re-enable ignoring the lock on the mutex above (this
            // information is lost in the lock tracking tables once a mutex is
            // unlocked).
            util::ignore_lock(&mtx_);

            return result;
        }

    private:
        mutable mutex_type mtx_;
    };
}}


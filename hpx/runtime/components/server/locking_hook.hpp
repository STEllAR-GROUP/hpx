//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_LOCKING_HOOK_OCT_17_2012_0732PM)
#define HPX_COMPONENTS_SERVER_LOCKING_HOOK_OCT_17_2012_0732PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/get_lva.hpp>
#include <hpx/util/scoped_unlock.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/coroutine/coroutine.hpp>
#include <hpx/lcos/local/spinlock.hpp>

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
        locking_hook() : base_type() {}

        template <typename Arg>
        locking_hook(Arg && arg)
          : base_type(std::forward<Arg>(arg))
        {}

        /// This is the hook implementation for decorate_action which locks
        /// the component ensuring that only one action is executed at a time
        /// for this component instance.
        template <typename F>
        static threads::thread_function_type
        decorate_action(naming::address::address_type lva, F && f)
        {
            return util::bind(
                util::one_shot(&locking_hook::thread_function),
                get_lva<this_component_type>::call(lva),
                util::placeholders::_1,
                base_type::decorate_action(lva, std::forward<F>(f)));
        }

    protected:
        typedef util::function_nonser<
            threads::thread_state_ex_enum(threads::thread_state_enum)
        > yield_decorator_type;

        struct undecorate_wrapper
        {
            ~undecorate_wrapper()
            {
                threads::get_self().undecorate_yield();
            }
        };

        // Execute the wrapped action. This locks the mutex ensuring a thread
        // safe action invocation.
        threads::thread_state_enum thread_function(
            threads::thread_state_ex_enum state,
            threads::thread_function_type f)
        {
            threads::thread_state_enum result = threads::unknown;

            // now lock the mutex and execute the action
            typename mutex_type::scoped_lock l(mtx_);

            {
                // register our yield decorator
                using util::placeholders::_1;
                threads::get_self().decorate_yield(
                    util::bind(&locking_hook::yield_function, this, _1));

                undecorate_wrapper yield_undecorator;
                (void)yield_undecorator;       // silence gcc warnings

                result = f(state);
            }

            return result;
        }

        struct decorate_wrapper
        {
            decorate_wrapper()
              : yield_decorator_(threads::get_self().undecorate_yield())
            {}

            ~decorate_wrapper()
            {
                threads::get_self().decorate_yield(std::move(yield_decorator_));
            }

            yield_decorator_type yield_decorator_;
        };

        // The yield decorator unlocks the mutex and calls the system yield
        // which gives up control back to the thread manager.
        threads::thread_state_ex_enum yield_function(
            threads::thread_state_enum state)
        {
            // We un-decorate the yield function as the lock handling may
            // suspend, which causes an infinite recursion otherwise.
            decorate_wrapper yield_decorator;
            threads::thread_state_ex_enum result = threads::wait_unknown;

            {
                util::scoped_unlock<mutex_type> ul(mtx_);
                result = threads::get_self().yield_impl(state);
            }

            return result;
        }

    private:
        mutable mutex_type mtx_;
    };
}}

#endif

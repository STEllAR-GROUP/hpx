//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_LOCKING_HOOK_OCT_17_2012_0732PM)
#define HPX_COMPONENTS_SERVER_LOCKING_HOOK_OCT_17_2012_0732PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/get_lva.hpp>
#include <hpx/util/unlock_lock.hpp>
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
        locking_hook(BOOST_FWD_REF(Arg) arg) 
          : base_type(boost::forward<Arg>(arg)) 
        {}

        /// This is the hook implementation for decorate_action which locks
        /// the component ensuring that only one action is executed at a time
        /// for this component instance.
        static HPX_STD_FUNCTION<threads::thread_function_type> 
        wrap_action(HPX_STD_FUNCTION<threads::thread_function_type> f,
            naming::address::address_type lva)
        {
            using HPX_STD_PLACEHOLDERS::_1;

            return HPX_STD_BIND(&locking_hook::thread_function, 
                get_lva<this_component_type>::call(lva), _1, 
                boost::move(base_type::wrap_action(boost::move(f), lva)));
        }

    protected:
        // Execute the wrapped action. This locks the mutex ensuring a thread 
        // safe action invocation.
        threads::thread_state_enum thread_function(
            threads::thread_state_ex_enum state, 
            HPX_STD_FUNCTION<threads::thread_function_type> f) 
        {
            using HPX_STD_PLACEHOLDERS::_1;

            // register our yield decorator
            threads::get_self().decorate_yield(
                HPX_STD_BIND(&locking_hook::yield_function, this, _1));

            // now lock the mutex and execute the action
            typename mutex_type::scoped_lock l(mtx_);
            return f(state);
        }

        // The yield decorator unlocks the mutex and calls the system yield 
        // which gives up control back to the thread manager.
        threads::thread_state_ex_enum yield_function(
            threads::thread_state_enum state) 
        {
            util::unlock_the_lock<mutex_type> ul(mtx_);
            return threads::get_self().yield_impl(state);
        }

    private:
        mutable mutex_type mtx_;
    };
}}

#endif

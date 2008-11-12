//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLIER_APPLY_HELPER_JUN_25_2008_0917PM)
#define HPX_APPLIER_APPLY_HELPER_JUN_25_2008_0917PM

#include <boost/mpl/bool.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/naming.hpp>
#include <hpx/runtime/actions/continuation.hpp>

namespace hpx { namespace applier { namespace detail
{
    ///////////////////////////////////////////////////////////////////////
    template <
        typename Action,
        typename DirectExecute = typename Action::direct_execution
    >
    struct apply_helper0;

    template <typename Action>
    struct apply_helper0<Action, boost::mpl::false_>
    {
        static void 
        call (Action* act, threads::threadmanager& tm, applier& appl, 
            naming::address::address_type lva)
        {
            tm.register_work(act->get_thread_function(appl, lva),
                actions::detail::get_action_name<Action>());
        }

        static void 
        call (threads::threadmanager& tm, applier& appl, 
            naming::address::address_type lva)
        {
            tm.register_work(Action::construct_thread_function(appl, lva),
                actions::detail::get_action_name<Action>());
        }

        static void 
        call (actions::continuation_type& c, 
            threads::threadmanager& tm, applier& appl, 
            naming::address::address_type lva)
        {
            tm.register_work(Action::construct_thread_function(c, appl, lva),
                actions::detail::get_action_name<Action>());
        }
    };

    template <typename Action>
    struct apply_helper0<Action, boost::mpl::true_>
    {
        static void 
        call (Action* act, threads::threadmanager&, applier& appl, 
            naming::address::address_type lva)
        {
            BOOST_ASSERT(false);    // shouldn't be called at all
        }

        // If local and to be directly executed, just call the function
        static void
        call (threads::threadmanager&, applier& appl, 
            naming::address::address_type addr)
        {
            Action::execute_function(appl, addr);
        }

        static typename Action::result_type 
        call (actions::continuation_type& c, threads::threadmanager&, 
            applier& appl, naming::address::address_type addr)
        {
            try {
                return c->trigger_all(appl, Action::execute_function(appl, addr));
            }
            catch (hpx::exception const& e) {
                // make sure hpx::exceptions are propagated back to the client
                c->trigger_error(appl, e);
                return typename Action::result_type();
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////
    template <
        typename Action, typename Arg0, 
        typename DirectExecute = typename Action::direct_execution
    >
    struct apply_helper1;

    template <typename Action, typename Arg0>
    struct apply_helper1<Action, Arg0, boost::mpl::false_>
    {
        static void 
        call (threads::threadmanager& tm, applier& appl, 
            naming::address::address_type addr, Arg0 const& arg0)
        {
            tm.register_work(Action::construct_thread_function(appl, addr, arg0),
                actions::detail::get_action_name<Action>());
        }

        static void 
        call (actions::continuation_type& c, 
            threads::threadmanager& tm, applier& appl, 
            naming::address::address_type addr, Arg0 const& arg0)
        {
            tm.register_work(Action::construct_thread_function(c, appl, addr, arg0),
                actions::detail::get_action_name<Action>());
        }
    };

    template <typename Action, typename Arg0>
    struct apply_helper1<Action, Arg0, boost::mpl::true_>
    {
        // If local and to be directly executed, just call the function
        static void
        call (threads::threadmanager&, applier& appl, 
            naming::address::address_type addr, Arg0 const& arg0)
        {
            Action::execute_function(appl, addr, arg0);
        }

        static typename Action::result_type  
        call (actions::continuation_type& c, threads::threadmanager&, 
            applier& appl, naming::address::address_type addr, 
            Arg0 const& arg0)
        {
            try {
                return c->trigger_all(appl, 
                    Action::execute_function(appl, addr, arg0));
            }
            catch (hpx::exception const& e) {
                // make sure hpx::exceptions are propagated back to the client
                c->trigger_error(appl, e);
                return typename Action::result_type();
            }
        }
    };

    // bring in the rest of the apply<> overloads
    #include <hpx/runtime/applier/apply_helper_implementations.hpp>

}}}

#endif

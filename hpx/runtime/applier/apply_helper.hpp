//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLIER_APPLY_HELPER_JUN_25_2008_0917PM)
#define HPX_APPLIER_APPLY_HELPER_JUN_25_2008_0917PM

#include <boost/mpl/bool.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>

namespace hpx { namespace applier { namespace detail
{
    ///////////////////////////////////////////////////////////////////////
    template <typename Action,
        typename DirectExecute = typename Action::direct_execution>
    struct apply_helper0;

    template <typename Action>
    struct apply_helper0<Action, boost::mpl::false_>
    {
        static void
        call (Action* act, naming::address::address_type lva,
            threads::thread_priority priority)
        {
            hpx::applier::register_work_plain(act->get_thread_function(lva),
                actions::detail::get_action_name<Action>(), lva,
                threads::pending, priority);
        }

        static void
        call (naming::address::address_type lva,
            threads::thread_priority priority)
        {
            hpx::applier::register_work_plain(
                Action::construct_thread_function(lva),
                actions::detail::get_action_name<Action>(), lva,
                threads::pending, priority);
        }

        static void
        call (actions::continuation_type& c, naming::address::address_type lva,
            threads::thread_priority priority)
        {
            hpx::applier::register_work_plain(
                Action::construct_thread_function(c, lva),
                actions::detail::get_action_name<Action>(), lva,
                threads::pending, priority);
        }
    };

    template <typename Action>
    struct apply_helper0<Action, boost::mpl::true_>
    {
        static void
        call (Action* act, naming::address::address_type lva,
            threads::thread_priority /*priority*/)
        {
            BOOST_ASSERT(false);    // shouldn't be called at all
        }

        // If local and to be directly executed, just call the function
        static void
        call (naming::address::address_type lva,
            threads::thread_priority /*priority*/)
        {
            Action::execute_function(lva);
        }

        static void
        call (actions::continuation_type& c, naming::address::address_type lva,
            threads::thread_priority /*priority*/)
        {
            try {
                c->trigger(boost::move(Action::execute_function(lva)));
            }
            catch (hpx::exception const& e) {
                // make sure hpx::exceptions are propagated back to the client
                c->trigger_error(boost::current_exception());
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename Action,
        typename DirectExecute = typename Action::direct_execution>
    struct apply_helper1;

    template <typename Action>
    struct apply_helper1<Action, boost::mpl::false_>
    {
        template <typename Arg0>
        static void
        call (naming::address::address_type lva,
            threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0)
        {
            hpx::applier::register_work_plain(
                boost::move(Action::construct_thread_function(lva, 
                    boost::forward<Arg0>(arg0))),
                actions::detail::get_action_name<Action>(), lva,
                threads::pending, priority);
        }

        template <typename Arg0>
        static void
        call (actions::continuation_type& c, naming::address::address_type lva,
            threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0)
        {
            hpx::applier::register_work_plain(
                boost::move(Action::construct_thread_function(c, lva, 
                    boost::forward<Arg0>(arg0))),
                actions::detail::get_action_name<Action>(), lva,
                threads::pending, priority);
        }
    };

    template <typename Action>
    struct apply_helper1<Action, boost::mpl::true_>
    {
        // If local and to be directly executed, just call the function
        template <typename Arg0>
        static void
        call (naming::address::address_type lva,
            threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0)
        {
            Action::execute_function(lva, boost::forward<Arg0>(arg0));
        }

        template <typename Arg0>
        static void
        call (actions::continuation_type& c, naming::address::address_type lva,
            threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0)
        {
            try {
                c->trigger(boost::move(Action::execute_function(lva,
                    boost::forward<Arg0>(arg0))));
            }
            catch (hpx::exception const& /*e*/) {
                // make sure hpx::exceptions are propagated back to the client
                c->trigger_error(boost::current_exception());
            }
        }
    };
}}}

// bring in the rest of the apply<> overloads
#include <hpx/runtime/applier/apply_helper_implementations.hpp>

#endif

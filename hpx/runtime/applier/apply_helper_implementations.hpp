//  Copyright (c) 2007-2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_APPLIER_APPLY_HELPER_IMPLEMENTATIONS_JUN_26_2008_0150PM)
#define HPX_APPLIER_APPLY_HELPER_IMPLEMENTATIONS_JUN_26_2008_0150PM

#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (2, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/applier/apply_helper_implementations.hpp"))                  \
    /**/

#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

    ///////////////////////////////////////////////////////////////////////
    template <
        typename Action,
        BOOST_PP_ENUM_PARAMS(N, typename Arg),
        typename DirectExecute = typename Action::direct_execution>
    struct BOOST_PP_CAT(apply_helper, N);

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    struct BOOST_PP_CAT(apply_helper, N)<
        Action, BOOST_PP_ENUM_PARAMS(N, Arg), boost::mpl::false_>
    {
        static void
        call (naming::address::address_type lva,
            threads::thread_priority priority,
            BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
        {
            hpx::applier::register_work_plain(
                Action::construct_thread_function(lva,
                    BOOST_PP_ENUM_PARAMS(N, arg)),
                actions::detail::get_action_name<Action>(), lva,
                threads::pending, priority);
        }

        static void
        call (actions::continuation_type& c, naming::address::address_type lva,
            threads::thread_priority priority,
            BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
        {
            hpx::applier::register_work_plain(
                Action::construct_thread_function(
                    c, lva, BOOST_PP_ENUM_PARAMS(N, arg)),
                actions::detail::get_action_name<Action>(), lva,
                threads::pending, priority);
        }
    };

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    struct BOOST_PP_CAT(apply_helper, N)<
        Action, BOOST_PP_ENUM_PARAMS(N, Arg), boost::mpl::true_>
    {
        // If local and to be directly executed, just call the function
        static void
        call (naming::address::address_type lva,
            threads::thread_priority priority,
            BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
        {
            Action::execute_function_nonvirt(lva, BOOST_PP_ENUM_PARAMS(N, arg));
        }

        static typename Action::result_type
        call (actions::continuation_type& c, naming::address::address_type lva,
            threads::thread_priority priority,
            BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
        {
            try {
                return c->trigger(Action::execute_function_nonvirt(
                    lva, BOOST_PP_ENUM_PARAMS(N, arg)));
            }
            catch (hpx::exception const& e) {
                // make sure hpx::exceptions are propagated back to the client
                c->trigger_error(boost::current_exception());
                return typename Action::result_type();
            }
        }
    };

#undef N

#endif

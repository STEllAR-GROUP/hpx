//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_APPLIER_APPLY_HELPER_IMPLEMENTATIONS_JUN_26_2008_0150PM)
#define HPX_APPLIER_APPLY_HELPER_IMPLEMENTATIONS_JUN_26_2008_0150PM

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
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

#define HPX_FWD_ARGS(z, n, _)                                                 \
        BOOST_PP_COMMA_IF(n)                                                  \
            BOOST_FWD_REF(BOOST_PP_CAT(Arg, n)) BOOST_PP_CAT(arg, n)          \
    /**/
#define HPX_FORWARD_ARGS(z, n, _)                                             \
        BOOST_PP_COMMA_IF(n)                                                  \
            boost::forward<BOOST_PP_CAT(Arg, n)>(BOOST_PP_CAT(arg, n))        \
    /**/

namespace hpx { namespace applier { namespace detail
{
    ///////////////////////////////////////////////////////////////////////
    template <typename Action,
        typename DirectExecute = typename Action::direct_execution>
    struct BOOST_PP_CAT(apply_helper, N);

    template <typename Action>
    struct BOOST_PP_CAT(apply_helper, N)<Action, boost::mpl::false_>
    {
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        static void
        call (naming::address::address_type lva,
            threads::thread_priority priority,
            BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            hpx::applier::register_work_plain(
                boost::move(Action::construct_thread_function(lva,
                    BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _))),
                actions::detail::get_action_name<Action>(), lva,
                threads::pending, priority);
        }

        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        static void
        call (actions::continuation_type& c, naming::address::address_type lva,
            threads::thread_priority priority,
            BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            hpx::applier::register_work_plain(
                boost::move(Action::construct_thread_function(
                    c, lva, BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _))),
                actions::detail::get_action_name<Action>(), lva,
                threads::pending, priority);
        }
    };

    template <typename Action>
    struct BOOST_PP_CAT(apply_helper, N)<Action, boost::mpl::true_>
    {
        // If local and to be directly executed, just call the function
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        static void
        call (naming::address::address_type lva,
            threads::thread_priority priority,
            BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            Action::execute_function(lva,
                BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        }

        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        static void
        call (actions::continuation_type& c, naming::address::address_type lva,
            threads::thread_priority priority,
            BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            try {
                c->trigger(boost::move(Action::execute_function(
                    lva, BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _))));
            }
            catch (hpx::exception const& e) {
                // make sure hpx::exceptions are propagated back to the client
                c->trigger_error(boost::current_exception());
            }
        }
    };
}}}

///////////////////////////////////////////////////////////////////////////////
#undef HPX_FORWARD_ARGS
#undef HPX_FWD_ARGS
#undef N

#endif

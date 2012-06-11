//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_TRANSFER_ACTIONS_ACTION_CONSTRUCTORS_MAY_05_2012_1018AM)
#define HPX_RUNTIME_TRANSFER_ACTIONS_ACTION_CONSTRUCTORS_MAY_05_2012_1018AM

#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (2, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/actions/transfer_action_constructors.hpp"))                  \
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

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    transfer_action(BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        : arguments_(BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _)),
          parent_locality_(transfer_action::get_locality_id()),
          parent_id_(reinterpret_cast<std::size_t>(threads::get_parent_id())),
          parent_phase_(threads::get_parent_phase()),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value))
    {}

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    transfer_action(threads::thread_priority priority,
              BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        : arguments_(BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _)),
          parent_locality_(transfer_action::get_locality_id()),
          parent_id_(reinterpret_cast<std::size_t>(threads::get_parent_id())),
          parent_phase_(threads::get_parent_phase()),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority))
    {}

#undef HPX_FORWARD_ARGS
#undef HPX_FWD_ARGS
#undef N

#endif

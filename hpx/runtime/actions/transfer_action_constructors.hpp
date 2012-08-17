//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_TRANSFER_ACTIONS_ACTION_CONSTRUCTORS_MAY_05_2012_1018AM)
#define HPX_RUNTIME_TRANSFER_ACTIONS_ACTION_CONSTRUCTORS_MAY_05_2012_1018AM

#include <hpx/util/move.hpp>

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/runtime/actions/preprocessed/transfer_action_constructors.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/transfer_action_constructors_" HPX_LIMIT_STR ".hpp")
#endif

#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (2, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/actions/transfer_action_constructors.hpp"))                  \
    /**/

#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    transfer_action(HPX_ENUM_FWD_ARGS(N, Arg, arg))
        : arguments_(HPX_ENUM_FORWARD_ARGS(N, Arg, arg)),
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
              HPX_ENUM_FWD_ARGS(N, Arg, arg))
        : arguments_(HPX_ENUM_FORWARD_ARGS(N, Arg, arg)),
          parent_locality_(transfer_action::get_locality_id()),
          parent_id_(reinterpret_cast<std::size_t>(threads::get_parent_id())),
          parent_phase_(threads::get_parent_phase()),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority))
    {}

#undef N

#endif

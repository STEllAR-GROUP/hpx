//  Copyright (c) 2007-2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_ACTIONS_ACTION_CONSTRUCTORS_MAY_20_2008_1045AM)
#define HPX_RUNTIME_ACTIONS_ACTION_CONSTRUCTORS_MAY_20_2008_1045AM

#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (2, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/actions/action_constructors.hpp"))                           \
    /**/

#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    action(BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
        : arguments_(BOOST_PP_ENUM_PARAMS(N, arg)),
          parent_locality_(applier::get_prefix_id()),
          parent_id_(reinterpret_cast<std::size_t>(threads::get_parent_id())),
          parent_phase_(threads::get_parent_phase()),
          priority_(detail::thread_priority<Priority>::call(Priority))
    {}

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    action(threads::thread_priority priority,
              BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
        : arguments_(BOOST_PP_ENUM_PARAMS(N, arg)),
          parent_locality_(applier::get_prefix_id()),
          parent_id_(reinterpret_cast<std::size_t>(threads::get_parent_id())),
          parent_phase_(threads::get_parent_phase()),
          priority_(detail::thread_priority<Priority>::call(priority))
    {}

#undef N

#endif

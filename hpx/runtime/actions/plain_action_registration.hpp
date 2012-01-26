
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//  Copyright (c) 2011      Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_ACTIONS_PLAIN_ACTION_REGISTRATION_NOV_06_2011_1404PM)
#define HPX_RUNTIME_ACTIONS_PLAIN_ACTION_REGISTRATION_NOV_06_2011_1404PM

#error "Automatic registration of plain_actions isn't supported yet!"

#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/util/detail/serialization_registration.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

///////////////////////////////////////////////////////////////////////////////
HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (template <typename Result, Result (*F)(),
        hpx::threads::thread_priority Priority, typename Derived>),
    (hpx::actions::plain_result_action0<Result, F, Priority, Derived>)
)

HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (template <typename Result, Result (*F)(), typename Derived>),
    (hpx::actions::plain_direct_result_action0<Result, F, Derived>)
)

HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (template <void (*F)(), hpx::threads::thread_priority Priority,
        typename Derived>),
    (hpx::actions::plain_action0<F, Priority, Derived>)
)

HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (template <void (*F)(), typename Derived>),
    (hpx::actions::plain_direct_action0<F, Derived>)
)

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/actions/plain_action_registration.hpp"))                     \
    /**/

#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

///////////////////////////////////////////////////////////////////////////////
HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (template <typename Result, BOOST_PP_ENUM_PARAMS(N, typename T),
        Result (*F)(BOOST_PP_ENUM_PARAMS(N, T)),
        hpx::threads::thread_priority Priority, typename Derived>),
    (BOOST_PP_CAT(hpx::actions::plain_result_action, N)<
        Result, BOOST_PP_ENUM_PARAMS(N, T), F, Priority, Derived>)
)

HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (template <typename Result, BOOST_PP_ENUM_PARAMS(N, typename T),
        Result (*F)(BOOST_PP_ENUM_PARAMS(N, T)), typename Derived>),
    (BOOST_PP_CAT(hpx::actions::plain_direct_result_action, N)<
        Result, BOOST_PP_ENUM_PARAMS(N, T), F, Derived>)
)

HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (template <BOOST_PP_ENUM_PARAMS(N, typename T),
        void (*F)(BOOST_PP_ENUM_PARAMS(N, T)),
        hpx::threads::thread_priority Priority, typename Derived>),
    (BOOST_PP_CAT(hpx::actions::plain_action, N)<
        BOOST_PP_ENUM_PARAMS(N, T), F, Priority, Derived>)
)

HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (template <BOOST_PP_ENUM_PARAMS(N, typename T),
        void (*F)(BOOST_PP_ENUM_PARAMS(N, T)), typename Derived>),
    (BOOST_PP_CAT(hpx::actions::plain_direct_action, N)<
        BOOST_PP_ENUM_PARAMS(N, T), F, Derived>)
)

#undef N
#endif

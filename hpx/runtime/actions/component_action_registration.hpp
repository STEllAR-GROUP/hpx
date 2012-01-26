
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//  Copyright (c) 2011      Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_ACTIONS_COMPONENT_ACTION_REGISTRATION_NOV_06_2011_1404PM)
#define HPX_RUNTIME_ACTIONS_COMPONENT_ACTION_REGISTRATION_NOV_06_2011_1404PM

#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/util/detail/serialization_registration.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

///////////////////////////////////////////////////////////////////////////////
// Register the action templates with serialization.
HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (
        template <
            typename Component
          , typename Result
          , int Action
          , Result (Component::*F)()
          , hpx::threads::thread_priority Priority
          , typename Derived
        >
    )
  , (
        hpx::actions::result_action0<Component, Result, Action, F, Priority, Derived>
    )
)

HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (
        template <
            typename Component
          , typename Result
          , int Action
          , Result (Component::*F)()
          , typename Derived
        >
    )
  , (
        hpx::actions::direct_result_action0<Component, Result, Action, F, Derived>
    )
)

HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (
        template <
            typename Component
          , int Action
          , void (Component::*F)()
          , hpx::threads::thread_priority Priority
          , typename Derived
        >
    )
  , (
        hpx::actions::action0<Component, Action, F, Priority, Derived>
    )
)

HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (
        template <
            typename Component
          , int Action
          , void (Component::*F)()
          , typename Derived
        >
    )
  , (
        hpx::actions::direct_action0<Component, Action, F, Derived>
    )
)

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/actions/component_action_registration.hpp"))                 \
    /**/

#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

///////////////////////////////////////////////////////////////////////////////
// Register the action templates with serialization.
HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (
        template <
            typename Component
          , typename Result
          , int Action
          , BOOST_PP_ENUM_PARAMS(N, typename T)
          , Result (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T))
          , hpx::threads::thread_priority Priority
          , typename Derived
        >
    )
  , (
        BOOST_PP_CAT(hpx::actions::result_action, N)<
            Component
          , Result
          , Action
          , BOOST_PP_ENUM_PARAMS(N, T)
          , F
          , Priority
          , Derived
        >
    )
)

HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (
        template <
            typename Component
          , typename Result
          , int Action
          , BOOST_PP_ENUM_PARAMS(N, typename T)
          , Result (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T))
          , typename Derived
        >
    )
  , (
        BOOST_PP_CAT(hpx::actions::direct_result_action, N)<
            Component
          , Result
          , Action
          , BOOST_PP_ENUM_PARAMS(N, T)
          , F
          , Derived
        >
    )
)

HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (
        template <
            typename Component
          , int Action
          , BOOST_PP_ENUM_PARAMS(N, typename T)
          , void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T))
          , hpx::threads::thread_priority Priority
          , typename Derived
        >
    )
  , (
        BOOST_PP_CAT(hpx::actions::action, N)<
            Component
          , Action
          , BOOST_PP_ENUM_PARAMS(N, T)
          , F
          , Priority
          , Derived
        >
    )
)

HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (
        template <
            typename Component
          , int Action
          , BOOST_PP_ENUM_PARAMS(N, typename T)
          , void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T))
          , typename Derived
        >
    )
  , (
        BOOST_PP_CAT(hpx::actions::direct_action, N)<
            Component
          , Action
          , BOOST_PP_ENUM_PARAMS(N, T)
          , F
          , Derived
        >
    )
)

#undef N
#endif

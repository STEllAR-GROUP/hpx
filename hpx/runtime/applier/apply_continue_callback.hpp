//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_APPLIER_APPLY_CONTINUE_CALLBACK_MAR_09_2014_1207PM)
#define HPX_RUNTIME_APPLIER_APPLY_CONTINUE_CALLBACK_MAR_09_2014_1207PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/applier/apply.hpp>

#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/runtime/applier/preprocessed/apply_continue_callback.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/apply_continue_callback_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (0, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/applier/apply_continue_callback.hpp"))                       \
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

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Action
      , typename Callback
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename T)
      , typename F>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      BOOST_PP_COMMA_IF(N) HPX_ENUM_FWD_ARGS(N, T, v)
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;

        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          BOOST_PP_COMMA_IF(N) HPX_ENUM_FORWARD_ARGS(N, T, v));
    }

    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename T)
      , typename F>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/
      , naming::id_type const& gid
      , Callback && cb
      BOOST_PP_COMMA_IF(N) HPX_ENUM_FWD_ARGS(N, T, v)
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          BOOST_PP_COMMA_IF(N) HPX_ENUM_FORWARD_ARGS(N, T, v)
          , std::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Action
      , typename Callback
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename T)>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      BOOST_PP_COMMA_IF(N) HPX_ENUM_FWD_ARGS(N, T, v)
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;

        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          BOOST_PP_COMMA_IF(N) HPX_ENUM_FORWARD_ARGS(N, T, v));
    }

    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename T)>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/
      , naming::id_type const& gid
      , Callback && cb
      BOOST_PP_COMMA_IF(N) HPX_ENUM_FWD_ARGS(N, T, v)
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          BOOST_PP_COMMA_IF(N) HPX_ENUM_FORWARD_ARGS(N, T, v)
          , cont);
    }
}

#undef N

#endif

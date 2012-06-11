//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_APPLY_APR_16_20012_0943AM)
#define HPX_APPLY_APR_16_20012_0943AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>
#include <hpx/traits/supports_result_of.hpp>

#include <boost/move/move.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/enum_params.hpp>
#include <boost/preprocessor/iterate.hpp>

///////////////////////////////////////////////////////////////////////////////
#define HPX_FWD_ARGS(z, n, _)                                                 \
            BOOST_FWD_REF(BOOST_PP_CAT(A, n)) BOOST_PP_CAT(a, n)              \
    /**/
#define HPX_FORWARD_ARGS(z, n, _)                                             \
            boost::forward<BOOST_PP_CAT(A, n)>(BOOST_PP_CAT(a, n))            \
    /**/

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    // simply launch the given function or function object asynchronously
    template <typename F>
    bool apply(BOOST_FWD_REF(F) f)
    {
        thread t(boost::forward<F>(f));
        t.detach();     // detach the thread
        return false;   // executed locally
    }

    ///////////////////////////////////////////////////////////////////////////
    // Define apply() overloads for plain local functions and function objects.
    //
    // Note that these overloads are limited to function objects supporting the
    // result_of protocol. We will need to revisit this as soon as we will be
    // able to implement a proper is_callable trait (current compiler support
    // does not allow to do that portably).

    // simply launch the given function or function object asynchronously

#define HPX_UTIL_BOUND_FUNCTION_APPLY(Z, N, D)                                \
    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename A)>                \
    typename boost::enable_if<                                                \
        traits::supports_result_of<F>                                         \
      , bool                                                                  \
    >::type                                                                   \
    apply(BOOST_FWD_REF(F) f, BOOST_PP_ENUM(N, HPX_FWD_ARGS, _))              \
    {                                                                         \
        thread t(boost::forward<F>(f),                                        \
            BOOST_PP_ENUM(N, HPX_FORWARD_ARGS, _));                           \
        t.detach();                                                           \
        return false;                                                         \
    }                                                                         \
    /**/

    BOOST_PP_REPEAT_FROM_TO(
        1
      , HPX_FUNCTION_LIMIT
      , HPX_UTIL_BOUND_FUNCTION_APPLY, _
    )

#undef HPX_UTIL_BOUND_FUNCTION_APPLY

}

///////////////////////////////////////////////////////////////////////////////
// bring in all N-nary overloads for apply
#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT, <hpx/apply.hpp>))                      \
    /**/

#include BOOST_PP_ITERATE()

#undef HPX_FORWARD_ARGS
#undef HPX_FWD_ARGS

///////////////////////////////////////////////////////////////////////////////
#else

#define N BOOST_PP_ITERATION()
#define NN BOOST_PP_ITERATION()

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    // apply a nullary bound function
    template <
        typename R
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename T)
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            BOOST_PP_CAT(hpx::util::detail::bound_function, N)<
                R
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, T)
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, Arg)
            >
        ))) bound)
    {
        thread t(boost::move(bound));
        t.detach();     // detach the thread
        return false;   // executed locally
    }

    // define apply() overloads for n-nary bound actions
#define HPX_UTIL_BOUND_FUNCTION_APPLY(Z, N, D)                                \
    template <                                                                \
        typename R                                                            \
      BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, typename T)              \
      BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, typename Arg)            \
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)                \
    >                                                                         \
    bool apply(                                                               \
        BOOST_RV_REF(HPX_UTIL_STRIP((                                         \
            BOOST_PP_CAT(hpx::util::detail::bound_function, NN)<              \
                R                                                             \
              BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, T)               \
              BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, Arg)             \
            >) bound                                                          \
      , BOOST_PP_ENUM(N, HPX_FWD_ARGS, _)                                     \
    )))                                                                       \
    {                                                                         \
        thread t(boost::move(bound)                                           \
          , BOOST_PP_ENUM(N, HPX_FORWARD_ARGS, _));                           \
        t.detach();                                                           \
        return false;                                                         \
    }                                                                         \
    /**/

    BOOST_PP_REPEAT_FROM_TO(
        1
      , HPX_FUNCTION_LIMIT
      , HPX_UTIL_BOUND_FUNCTION_APPLY, _
    )

#undef HPX_UTIL_BOUND_FUNCTION_APPLY

    ///////////////////////////////////////////////////////////////////////////
    // apply a nullary bound member function
    template <
        typename R
      , typename C
      BOOST_PP_COMMA_IF(BOOST_PP_DEC(N))
            BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), typename T)
      BOOST_PP_COMMA_IF(N)  BOOST_PP_ENUM_PARAMS(N, typename Arg)
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            BOOST_PP_CAT(hpx::util::detail::bound_member_function, N)<
                R
              , C
              BOOST_PP_COMMA_IF(BOOST_PP_DEC(N))
                    BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), T)
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, Arg)
            >
        ))) bound)
    {
        thread t(boost::move(bound));
        t.detach();     // detach the thread
        return false;   // executed locally
    }

    // define apply() overloads for n-nary bound actions
#define HPX_UTIL_BOUND_MEMBER_FUNCTION_APPLY(Z, N, D)                         \
    template <                                                                \
        typename R                                                            \
      , typename C                                                            \
      BOOST_PP_COMMA_IF(BOOST_PP_DEC(NN))                                     \
          BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(NN), typename T)                  \
      BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, typename Arg)            \
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)                \
    >                                                                         \
    bool apply(                                                               \
        BOOST_RV_REF(HPX_UTIL_STRIP((                                         \
            BOOST_PP_CAT(hpx::util::detail::bound_member_function, NN)<       \
                R                                                             \
              , C                                                             \
              BOOST_PP_COMMA_IF(BOOST_PP_DEC(NN))                             \
                    BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(NN), T)                 \
              BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, Arg)             \
            >) bound                                                          \
      , BOOST_PP_ENUM(N, HPX_FWD_ARGS, _)                                     \
    )))                                                                       \
    {                                                                         \
        thread t(boost::move(bound)                                           \
          , BOOST_PP_ENUM(N, HPX_FORWARD_ARGS, _));                           \
        t.detach();                                                           \
        return false;                                                         \
    }                                                                         \
    /**/

    BOOST_PP_REPEAT_FROM_TO(
        1
      , HPX_FUNCTION_LIMIT
      , HPX_UTIL_BOUND_MEMBER_FUNCTION_APPLY, _
    )

#undef HPX_UTIL_BOUND_MEMBER_FUNCTION_APPLY

    ///////////////////////////////////////////////////////////////////////////
    // apply a nullary bound function object
    template <
        typename F
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            BOOST_PP_CAT(hpx::util::detail::bound_functor, N)<
                F
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, Arg)
            >
        ))) bound)
    {
        thread t(boost::move(bound));
        t.detach();     // detach the thread
        return false;   // executed locally
    }

    // define apply() overloads for n-nary bound actions
#define HPX_UTIL_BOUND_FUNCTOR_APPLY(Z, N, D)                                 \
    template <                                                                \
        typename F                                                            \
      BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, typename Arg)            \
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)                \
    >                                                                         \
    bool apply(                                                               \
        BOOST_RV_REF(HPX_UTIL_STRIP((                                         \
            BOOST_PP_CAT(hpx::util::detail::bound_functor, NN)<               \
                F                                                             \
              BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, Arg)             \
            >) bound                                                          \
      , BOOST_PP_ENUM(N, HPX_FWD_ARGS, _)                                     \
    )))                                                                       \
    {                                                                         \
        thread t(boost::move(bound), BOOST_PP_ENUM(N, HPX_FORWARD_ARGS, _));  \
        t.detach();                                                           \
        return false;                                                         \
    }                                                                         \
    /**/

    BOOST_PP_REPEAT_FROM_TO(
        1
      , HPX_FUNCTION_LIMIT
      , HPX_UTIL_BOUND_FUNCTOR_APPLY, _
    )

#undef HPX_UTIL_BOUND_FUNCTOR_APPLY

    ///////////////////////////////////////////////////////////////////////////
    // apply a nullary bound action
    template <
        typename Action
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename T)
    >
    bool apply(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            BOOST_PP_CAT(hpx::util::detail::bound_action, N)<
                Action
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, T)
            >))) bound)
    {
        return bound.apply();
    }

    // define apply() overloads for n-nary bound actions
#define HPX_UTIL_BOUND_ACTION_APPLY(Z, N, D)                                  \
    template <                                                                \
        typename Action                                                       \
      BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, typename T)              \
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)                \
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)                \
    >                                                                         \
    bool apply(                                                               \
        BOOST_RV_REF(HPX_UTIL_STRIP((                                        \
            BOOST_PP_CAT(hpx::util::detail::bound_action, NN)<                \
                Action                                                        \
              BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, T)               \
            >))) bound                                                        \
      , BOOST_PP_ENUM(N, HPX_FWD_ARGS, _)                                     \
    )                                                                         \
    {                                                                         \
        return bound.apply(BOOST_PP_ENUM(N, HPX_FORWARD_ARGS, _));            \
    }                                                                         \
    /**/

    BOOST_PP_REPEAT_FROM_TO(
        1
      , HPX_FUNCTION_LIMIT
      , HPX_UTIL_BOUND_ACTION_APPLY, _
    )

#undef HPX_UTIL_BOUND_ACTION_APPLY

}

#undef NN
#undef N

#endif


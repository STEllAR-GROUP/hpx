//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_ASYNC_APR_16_20012_0225PM)
#define HPX_ASYNC_APR_16_20012_0225PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/bind_action.hpp>
#include <hpx/util/protect.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>
#include <hpx/traits/supports_result_of.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/utility/result_of.hpp>
#include <boost/type_traits/remove_reference.hpp>

#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/enum_params.hpp>
#include <boost/preprocessor/iterate.hpp>

#if !defined(HPX_DONT_USE_PREPROCESSED_FILES)
#  include <hpx/preprocessed/async.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/async_" HPX_LIMIT_STR ".hpp")
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    namespace detail
    {
        // Defer the evaluation of result_of during the SFINAE checks below
        template <typename F, typename ResultOf = boost::result_of<F> >
        struct create_future
        {
            typedef lcos::future<typename ResultOf::type> type;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // Launch the given function or function object asynchronously and return a
    // future allowing to synchronize with the returned result.
    template <typename F>
    typename detail::create_future<F()>::type
    async (BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f)
    {
        typedef typename boost::result_of<F()>::type result_type;
        lcos::local::futures_factory<result_type()> p(
            boost::forward<F>(f));
        if (policy & launch::async)
            p.apply();
        return p.get_future();
    }

    template <typename F>
    typename detail::create_future<F()>::type
    async (BOOST_FWD_REF(F) f)
    {
        return async(launch::all, boost::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    // Define async() overloads for plain local functions and function objects.
    //
    // Note that these overloads are limited to function objects supporting the
    // result_of protocol. We will need to revisit this as soon as we will be
    // able to implement a proper is_callable trait (current compiler support
    // does not allow to do that portably).

    // Launch the given function or function object asynchronously and return a
    // future allowing to synchronize with the returned result.

#define HPX_UTIL_BOUND_FUNCTION_ASYNC(Z, N, D)                                \
    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename A)>                \
    typename boost::lazy_enable_if<                                           \
        traits::supports_result_of<F>                                         \
      , detail::create_future<                                                \
            typename boost::remove_reference<F>::type(                        \
                BOOST_PP_ENUM_PARAMS(N, A)                                    \
            )                                                                 \
        >                                                                     \
    >::type                                                                   \
    async (BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f,              \
        HPX_ENUM_FWD_ARGS(N, A, a))                                           \
    {                                                                         \
        typedef typename boost::result_of<                                    \
            F(BOOST_PP_ENUM_PARAMS(N, A))                                     \
        >::type result_type;                                                  \
        lcos::local::futures_factory<result_type()> p(                        \
            util::bind(boost::forward<F>(f),                                  \
                HPX_ENUM_FORWARD_ARGS(N, A, a)));                             \
        if (policy & launch::async) p.apply();                                \
        return p.get_future();                                                \
    }                                                                         \
    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename A)>                \
    typename boost::lazy_enable_if<                                           \
        traits::supports_result_of<F>                                         \
      , detail::create_future<                                                \
            typename boost::remove_reference<F>::type(                        \
                BOOST_PP_ENUM_PARAMS(N, A)                                    \
            )                                                                 \
        >                                                                     \
    >::type                                                                   \
    async (BOOST_FWD_REF(F) f, HPX_ENUM_FWD_ARGS(N, A, a))                    \
    {                                                                         \
        return async(launch::all, boost::forward<F>(f),                       \
            HPX_ENUM_FORWARD_ARGS(N, A, a));                                  \
    }                                                                         \
    /**/

    BOOST_PP_REPEAT_FROM_TO(
        1
      , HPX_FUNCTION_ARGUMENT_LIMIT
      , HPX_UTIL_BOUND_FUNCTION_ASYNC, _
    )

#undef HPX_UTIL_BOUND_FUNCTION_ASYNC

}

///////////////////////////////////////////////////////////////////////////////
// bring in all N-nary overloads for async
#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT, <hpx/async.hpp>))                      \
    /**/

#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_DONT_USE_PREPROCESSED_FILES)

#endif

///////////////////////////////////////////////////////////////////////////////
#else

#define N BOOST_PP_ITERATION()
#define NN BOOST_PP_ITERATION()

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    // Invoke a nullary bound function asynchronously and return a future
    // allowing to synchronize with the returned result.
    template <
        typename R
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename T)
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)
    >
    lcos::future<R>
    async(BOOST_SCOPED_ENUM(launch) policy,
        BOOST_RV_REF(HPX_UTIL_STRIP((
            BOOST_PP_CAT(hpx::util::detail::bound_function, N)<
                R
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, T)
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, Arg)
            >
        ))) bound)
    {
        lcos::local::futures_factory<R()> p(boost::move(bound));
        if (policy & launch::async)
            p.apply();
        return p.get_future();
    }

    template <
        typename R
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename T)
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)
    >
    lcos::future<R>
    async(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            BOOST_PP_CAT(hpx::util::detail::bound_function, N)<
                R
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, T)
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, Arg)
            >
        ))) bound)
    {
        return async(launch::all, boost::move(bound));
    }

    // define apply() overloads for n-nary bound actions
#define HPX_UTIL_BOUND_FUNCTION_ASYNC(Z, N, D)                                \
    template <                                                                \
        typename R                                                            \
      BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, typename T)              \
      BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, typename Arg)            \
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)                \
    >                                                                         \
    lcos::future<R>                                                           \
    async(BOOST_SCOPED_ENUM(launch) policy,                                   \
        BOOST_RV_REF(HPX_UTIL_STRIP((                                         \
            BOOST_PP_CAT(hpx::util::detail::bound_function, NN)<              \
                R                                                             \
              BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, T)               \
              BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, Arg)             \
            >))) bound                                                        \
      , HPX_ENUM_FWD_ARGS(N, A, a)                                            \
    )                                                                         \
    {                                                                         \
        lcos::local::futures_factory<R()> p(                                  \
            util::bind(                                                       \
                util::protect(boost::move(bound))                             \
              , HPX_ENUM_FORWARD_ARGS(N, A, a)));                             \
        if (policy & launch::async) p.apply();                                \
        return p.get_future();                                                \
    }                                                                         \
    template <                                                                \
        typename R                                                            \
      BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, typename T)              \
      BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, typename Arg)            \
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)                \
    >                                                                         \
    lcos::future<R>                                                           \
    async(                                                                    \
        BOOST_RV_REF(HPX_UTIL_STRIP((                                         \
            BOOST_PP_CAT(hpx::util::detail::bound_function, NN)<              \
                R                                                             \
              BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, T)               \
              BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, Arg)             \
            >))) bound                                                        \
      , HPX_ENUM_FWD_ARGS(N, A, a)                                            \
    )                                                                         \
    {                                                                         \
        return async(launch::all, boost::move(bound),                         \
            HPX_ENUM_FORWARD_ARGS(N, A, a));                                  \
    }                                                                         \
    /**/

    BOOST_PP_REPEAT_FROM_TO(
        1
      , HPX_FUNCTION_ARGUMENT_LIMIT
      , HPX_UTIL_BOUND_FUNCTION_ASYNC, _
    )

#undef HPX_UTIL_BOUND_FUNCTION_ASYNC

    ///////////////////////////////////////////////////////////////////////////
    // Invoke a nullary bound member function asynchronously and return a future
    // allowing to synchronize with the returned result.
    template <
        typename R
      , typename C
      BOOST_PP_COMMA_IF(BOOST_PP_DEC(N))
            BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), typename T)
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)
    >
    lcos::future<R>
    async(BOOST_SCOPED_ENUM(launch) policy,
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
        lcos::local::futures_factory<R()> p(boost::move(bound));
        if (policy & launch::async)
            p.apply();
        return p.get_future();
    }

    template <
        typename R
      , typename C
      BOOST_PP_COMMA_IF(BOOST_PP_DEC(N))
            BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), typename T)
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)
    >
    lcos::future<R>
    async(
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
        return async(launch::all, boost::move(bound));
    }

    // define async() overloads for n-nary bound member functions
#define HPX_UTIL_BOUND_MEMBER_FUNCTION_ASYNC(Z, N, D)                         \
    template <                                                                \
        typename R                                                            \
      , typename C                                                            \
      BOOST_PP_COMMA_IF(BOOST_PP_DEC(NN))                                     \
          BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(NN), typename T)                  \
      BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, typename Arg)            \
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)                \
    >                                                                         \
    lcos::future<R>                                                           \
    async(BOOST_SCOPED_ENUM(launch) policy,                                   \
        BOOST_RV_REF(HPX_UTIL_STRIP((                                         \
            BOOST_PP_CAT(hpx::util::detail::bound_member_function, NN)<       \
                R                                                             \
              , C                                                             \
              BOOST_PP_COMMA_IF(BOOST_PP_DEC(NN))                             \
                    BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(NN), T)                 \
              BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, Arg)             \
            >))) bound                                                        \
      , HPX_ENUM_FWD_ARGS(N, A, a)                                            \
    )                                                                         \
    {                                                                         \
        lcos::local::futures_factory<R()> p(                                  \
            util::bind(                                                       \
                util::protect(boost::move(bound))                             \
              , HPX_ENUM_FORWARD_ARGS(N, A, a)));                             \
        if (policy & launch::async) p.apply();                                \
        return p.get_future();                                                \
    }                                                                         \
    template <                                                                \
        typename R                                                            \
      , typename C                                                            \
      BOOST_PP_COMMA_IF(BOOST_PP_DEC(NN))                                     \
          BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(NN), typename T)                  \
      BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, typename Arg)            \
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)                \
    >                                                                         \
    lcos::future<R>                                                           \
    async(                                                                    \
        BOOST_RV_REF(HPX_UTIL_STRIP((                                         \
            BOOST_PP_CAT(hpx::util::detail::bound_member_function, NN)<       \
                R                                                             \
              , C                                                             \
              BOOST_PP_COMMA_IF(BOOST_PP_DEC(NN))                             \
                    BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(NN), T)                 \
              BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, Arg)             \
            >))) bound                                                        \
      , HPX_ENUM_FWD_ARGS(N, A, a)                                            \
    )                                                                         \
    {                                                                         \
        return async(launch::all, boost::move(bound),                         \
            HPX_ENUM_FORWARD_ARGS(N, A, a));                                  \
    }                                                                         \
    /**/

    BOOST_PP_REPEAT_FROM_TO(
        1
      , HPX_FUNCTION_ARGUMENT_LIMIT
      , HPX_UTIL_BOUND_MEMBER_FUNCTION_ASYNC, _
    )

#undef HPX_UTIL_BOUND_MEMBER_FUNCTION_ASYNC

    ///////////////////////////////////////////////////////////////////////////
    // Invoke a nullary function object asynchronously and return a future
    // allowing to synchronize with the returned result.
    template <
        typename F
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)
    >
    typename detail::create_future<F()>::type
    async(BOOST_SCOPED_ENUM(launch) policy,
        BOOST_RV_REF(HPX_UTIL_STRIP((
            BOOST_PP_CAT(hpx::util::detail::bound_functor, N)<
                F
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, Arg)
            >
        ))) bound)
    {
        typedef typename detail::create_future<F()>::type result_type;
        lcos::local::futures_factory<typename result_type::result_type()>
            p(boost::move(bound));
        if (policy & launch::async)
            p.apply();
        return p.get_future();
    }

    template <
        typename F
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)
    >
    typename detail::create_future<F()>::type
    async(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            BOOST_PP_CAT(hpx::util::detail::bound_functor, N)<
                F
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, Arg)
            >
        ))) bound)
    {
        return async(launch::all, boost::move(bound));
    }

    // define async() overloads for n-nary bound member functions
#define HPX_UTIL_BOUND_MEMBER_FUNCTOR_ASYNC(Z, N, D)                          \
    template <                                                                \
        typename F                                                            \
      BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, typename Arg)            \
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)                \
    >                                                                         \
    typename detail::create_future<                                           \
        F(BOOST_PP_ENUM_PARAMS(N, A))                                         \
    >::type                                                                   \
    async(BOOST_SCOPED_ENUM(launch) policy,                                   \
        BOOST_RV_REF(HPX_UTIL_STRIP((                                         \
            BOOST_PP_CAT(hpx::util::detail::bound_functor, NN)<               \
                F                                                             \
              BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, Arg)             \
            >))) bound                                                        \
      , HPX_ENUM_FWD_ARGS(N, A, a)                                            \
    )                                                                         \
    {                                                                         \
        typedef typename detail::create_future<                               \
                F(BOOST_PP_ENUM_PARAMS(N, A))                                 \
            >::type result_type;                                              \
        lcos::local::futures_factory<typename result_type::result_type()> p(  \
            util::bind(                                                       \
                util::protect(boost::move(bound))                             \
              , HPX_ENUM_FORWARD_ARGS(N, A, a)));                             \
        if (policy & launch::async) p.apply();                                \
        return p.get_future();                                                \
    }                                                                         \
    template <                                                                \
        typename F                                                            \
      BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, typename Arg)            \
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)                \
    >                                                                         \
    typename detail::create_future<                                           \
        F(BOOST_PP_ENUM_PARAMS(N, A))                                         \
    >::type                                                                   \
    async(                                                                    \
        BOOST_RV_REF(HPX_UTIL_STRIP((                                         \
            BOOST_PP_CAT(hpx::util::detail::bound_functor, NN)<               \
                F                                                             \
              BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, Arg)             \
            >))) bound                                                        \
      , HPX_ENUM_FWD_ARGS(N, A, a)                                            \
    )                                                                         \
    {                                                                         \
        return async(launch::all, boost::move(bound),                         \
            HPX_ENUM_FORWARD_ARGS(N, A, a));                                  \
    }                                                                         \
    /**/

    BOOST_PP_REPEAT_FROM_TO(
        1
      , HPX_FUNCTION_ARGUMENT_LIMIT
      , HPX_UTIL_BOUND_MEMBER_FUNCTOR_ASYNC, _
    )

#undef HPX_UTIL_BOUND_MEMBER_FUNCTOR_ASYNC

    ///////////////////////////////////////////////////////////////////////////
    // Launch the given bound action asynchronously and return a future
    // allowing to synchronize with the returned result.
    template <
        typename Action
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)
    >
    lcos::future<
        typename BOOST_PP_CAT(hpx::util::detail::bound_action, N)<
            Action
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, Arg)
        >::result_type
    >
    async(BOOST_SCOPED_ENUM(launch) policy,
        BOOST_RV_REF(HPX_UTIL_STRIP((
            BOOST_PP_CAT(hpx::util::detail::bound_action, N)<
                Action
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, Arg)
            >))) bound)
    {
        return bound.async();
    }

    template <
        typename Action
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)
    >
    lcos::future<
        typename BOOST_PP_CAT(hpx::util::detail::bound_action, N)<
            Action
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, Arg)
        >::result_type
    >
    async(
        BOOST_RV_REF(HPX_UTIL_STRIP((
            BOOST_PP_CAT(hpx::util::detail::bound_action, N)<
                Action
              BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, Arg)
            >))) bound)
    {
        return async(launch::all, boost::move(bound));
    }

    // define n-nary overloads
#define HPX_UTIL_BOUND_ACTION_ASYNC(Z, N, D)                                  \
    template <                                                                \
        typename Action                                                       \
      BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, typename Arg)            \
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)                \
    >                                                                         \
    lcos::future<                                                             \
        typename BOOST_PP_CAT(hpx::util::detail::bound_action, NN)<           \
            Action                                                            \
          BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, Arg)                 \
        >::result_type                                                        \
    >                                                                         \
    async(BOOST_SCOPED_ENUM(launch) policy,                                   \
        BOOST_RV_REF(HPX_UTIL_STRIP((                                         \
            BOOST_PP_CAT(hpx::util::detail::bound_action, NN)<                \
                Action                                                        \
              BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, Arg)             \
            >))) bound                                                        \
      , HPX_ENUM_FWD_ARGS(N, A, a)                                            \
    )                                                                         \
    {                                                                         \
        return bound.async(HPX_ENUM_FORWARD_ARGS(N, A, a));                   \
    }                                                                         \
    template <                                                                \
        typename Action                                                       \
      BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, typename Arg)            \
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)                \
    >                                                                         \
    lcos::future<                                                             \
        typename BOOST_PP_CAT(hpx::util::detail::bound_action, NN)<           \
            Action                                                            \
          BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, Arg)                 \
        >::result_type                                                        \
    >                                                                         \
    async(                                                                    \
        BOOST_RV_REF(HPX_UTIL_STRIP((                                         \
            BOOST_PP_CAT(hpx::util::detail::bound_action, NN)<                \
                Action                                                        \
              BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, Arg)             \
            >))) bound                                                        \
      , HPX_ENUM_FWD_ARGS(N, A, a)                                            \
    )                                                                         \
    {                                                                         \
        return async(launch::all, boost::move(bound),                         \
            HPX_ENUM_FORWARD_ARGS(N, A, a));                                  \
    }                                                                         \
    /**/

    BOOST_PP_REPEAT_FROM_TO(
        1
      , HPX_FUNCTION_ARGUMENT_LIMIT
      , HPX_UTIL_BOUND_ACTION_ASYNC, _
    )

#undef HPX_UTIL_BOUND_ACTION_ASYNC

}

#undef NN
#undef N

#endif


//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_ASYNC_APR_16_20012_0225PM)
#define HPX_ASYNC_APR_16_20012_0225PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/lcos/async_continue.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/util/bind_action.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/protect.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/utility/result_of.hpp>
#include <boost/type_traits/remove_reference.hpp>

#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/enum_params.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/type_traits/is_void.hpp>

namespace hpx { namespace detail
{
    // Defer the evaluation of result_of during the SFINAE checks below
#if defined(__clang__)
    template <typename F, typename Result = typename boost::result_of<F>::type>
    struct create_future
    {
        typedef lcos::future<Result> type;
    };
#elif _MSC_VER >= 1700
    // VS2012 has a decent implementation of std::result_of<>
    template <typename F, typename ResultOf = std::result_of<F> >
    struct create_future
    {
        typedef lcos::future<typename ResultOf::type> type;
    };
#else
    template <typename F, typename ResultOf = boost::result_of<F> >
    struct create_future
    {
        typedef lcos::future<typename ResultOf::type> type;
    };
#endif

    template <typename F>
    BOOST_FORCEINLINE typename detail::create_future<F()>::type
    call_sync(BOOST_FWD_REF(F) f, boost::mpl::false_)
    {
        return lcos::make_ready_future(f());
    }

    template <typename F>
    BOOST_FORCEINLINE typename detail::create_future<F()>::type
    call_sync(BOOST_FWD_REF(F) f, boost::mpl::true_)
    {
        f();
        return lcos::make_ready_future();
    }
}}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/preprocessed/async.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/async_" HPX_LIMIT_STR ".hpp")
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    // Launch the given function or function object asynchronously and return a
    // future allowing to synchronize with the returned result.
    template <typename F>
    typename boost::lazy_enable_if_c<
        !traits::is_action<typename util::decay<F>::type>::value
      , detail::create_future<F()>
    >::type
    async (BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f)
    {
        typedef typename boost::result_of<F()>::type result_type;
        if (policy == launch::sync)
        {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(boost::forward<F>(f), predicate());
        }

        lcos::local::futures_factory<result_type()> p(
            boost::forward<F>(f));
        if (detail::has_async_policy(policy))
            p.apply();
        return p.get_future();
    }

    template <typename F>
    typename boost::lazy_enable_if_c<
        !traits::is_action<typename util::decay<F>::type>::value
      , detail::create_future<F()>
    >::type
    async (threads::executor& sched, BOOST_FWD_REF(F) f)
    {
        typedef typename boost::result_of<F()>::type result_type;

        lcos::local::futures_factory<result_type()> p(sched,
            boost::forward<F>(f));
        p.apply();
        return p.get_future();
    }

    template <typename F>
    typename boost::lazy_enable_if_c<
        !traits::is_action<typename util::decay<F>::type>::value
      , detail::create_future<F()>
    >::type
    async (BOOST_FWD_REF(F) f)
    {
        return async(launch::all, boost::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    // Define async() overloads for plain local functions and function objects.

    // Launch the given function or function object asynchronously and return a
    // future allowing to synchronize with the returned result.

#define HPX_UTIL_BOUND_FUNCTION_ASYNC(Z, N, D)                                \
    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename A)>                \
    typename boost::lazy_enable_if_c<                                         \
        !traits::is_action<typename util::decay<F>::type>::value              \
      , detail::create_future<F(BOOST_PP_ENUM_PARAMS(N, A))>                  \
    >::type                                                                   \
    async (BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f,              \
        HPX_ENUM_FWD_ARGS(N, A, a))                                           \
    {                                                                         \
        typedef typename boost::result_of<                                    \
            F(BOOST_PP_ENUM_PARAMS(N, A))                                     \
        >::type result_type;                                                  \
        if (policy == launch::sync) {                                         \
            typedef typename boost::is_void<result_type>::type predicate;     \
            return detail::call_sync(util::bind(                              \
                util::protect(boost::forward<F>(f)),                          \
                HPX_ENUM_FORWARD_ARGS(N, A, a)), predicate());                \
        }                                                                     \
        lcos::local::futures_factory<result_type()> p(                        \
            util::bind(util::protect(boost::forward<F>(f)),                   \
                HPX_ENUM_FORWARD_ARGS(N, A, a)));                             \
        if (detail::has_async_policy(policy))                                 \
            p.apply();                                                        \
        return p.get_future();                                                \
    }                                                                         \
    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename A)>                \
    typename boost::lazy_enable_if_c<                                         \
        !traits::is_action<typename util::decay<F>::type>::value              \
      , detail::create_future<F(BOOST_PP_ENUM_PARAMS(N, A))>                  \
    >::type                                                                   \
    async (threads::executor& sched, BOOST_FWD_REF(F) f,                      \
        HPX_ENUM_FWD_ARGS(N, A, a))                                           \
    {                                                                         \
        typedef typename boost::result_of<F(BOOST_PP_ENUM_PARAMS(N, A))>::type\
            result_type;                                                      \
        lcos::local::futures_factory<result_type()> p(sched,                  \
            util::bind(util::protect(boost::forward<F>(f)),                   \
                HPX_ENUM_FORWARD_ARGS(N, A, a)));                             \
        p.apply();                                                            \
        return p.get_future();                                                \
    }                                                                         \
    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename A)>                \
    typename boost::lazy_enable_if_c<                                         \
        !traits::is_action<typename util::decay<F>::type>::value              \
      , detail::create_future<F(BOOST_PP_ENUM_PARAMS(N, A))>                  \
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

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

///////////////////////////////////////////////////////////////////////////////
#else

#define N BOOST_PP_ITERATION()
#define NN BOOST_PP_ITERATION()

namespace hpx
{

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
    async(threads::executor& sched,
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
    async(threads::executor& sched,                                           \
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


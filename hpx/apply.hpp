//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_APPLY_APR_16_20012_0943AM)
#define HPX_APPLY_APR_16_20012_0943AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/async.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/runtime/applier/apply_continue.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/bind_action.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>

#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/enum_params.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/utility/enable_if.hpp>

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/preprocessed/apply.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/apply_" HPX_LIMIT_STR ".hpp")
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    // simply launch the given function or function object asynchronously
    template <typename F>
    bool apply(threads::executor& sched, BOOST_FWD_REF(F) f)
    {
        sched.add(boost::forward<F>(f), "hpx::apply");
        return false;   // executed locally
    }

    template <typename F>
    bool apply(BOOST_FWD_REF(F) f)
    {
        threads::register_thread(boost::forward<F>(f), "hpx::apply");
        return false;   // executed locally
    }

    ///////////////////////////////////////////////////////////////////////////
    // Define apply() overloads for plain local functions and function objects.

    // simply launch the given function or function object asynchronously

#define HPX_UTIL_BOUND_FUNCTION_APPLY(Z, N, D)                                \
    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename A)>                \
    typename boost::enable_if_c<                                              \
        !traits::is_action<typename util::decay<F>::type>::value              \
      , bool                                                                  \
    >::type                                                                   \
    apply(threads::executor& sched, BOOST_FWD_REF(F) f,                       \
        HPX_ENUM_FWD_ARGS(N, A, a))                                           \
    {                                                                         \
        sched.add(util::bind(util::protect(boost::forward<F>(f)),             \
            HPX_ENUM_FORWARD_ARGS(N, A, a)), "hpx::apply");                   \
        return false;                                                         \
    }                                                                         \
    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename A)>                \
    typename boost::enable_if_c<                                              \
        !traits::is_action<typename util::decay<F>::type>::value              \
      , bool                                                                  \
    >::type                                                                   \
    apply(BOOST_FWD_REF(F) f, HPX_ENUM_FWD_ARGS(N, A, a))                     \
    {                                                                         \
        threads::register_thread(util::bind(                                  \
            util::protect(boost::forward<F>(f)),                              \
            HPX_ENUM_FORWARD_ARGS(N, A, a)), "hpx::apply");                   \
        return false;                                                         \
    }                                                                         \
    /**/

    BOOST_PP_REPEAT_FROM_TO(
        1
      , HPX_FUNCTION_ARGUMENT_LIMIT
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
    >                                                                         \
    bool apply(                                                               \
        BOOST_RV_REF(HPX_UTIL_STRIP((                                         \
            BOOST_PP_CAT(hpx::util::detail::bound_action, NN)<                \
                Action                                                        \
              BOOST_PP_COMMA_IF(NN) BOOST_PP_ENUM_PARAMS(NN, T)               \
            >))) bound                                                        \
      , HPX_ENUM_FWD_ARGS(N, A, a)                                            \
    )                                                                         \
    {                                                                         \
        return bound.apply(HPX_ENUM_FORWARD_ARGS(N, A, a));                   \
    }                                                                         \
    /**/

    BOOST_PP_REPEAT_FROM_TO(
        1
      , HPX_FUNCTION_ARGUMENT_LIMIT
      , HPX_UTIL_BOUND_ACTION_APPLY, _
    )

#undef HPX_UTIL_BOUND_ACTION_APPLY

}

#undef NN
#undef N

#endif


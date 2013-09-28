//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_UTIL_INVOKE_FUSED_HPP
#define HPX_UTIL_INVOKE_FUSED_HPP

#include <hpx/config.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>

#include <boost/type_traits/add_const.hpp>

namespace hpx { namespace util
{
    template <typename R, typename F>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f, util::tuple<>)
    {
        return invoke_r<R>(boost::forward<F>(f));
    }

    template <typename F>
    BOOST_FORCEINLINE
    typename invoke_result_of<F()>::type
    invoke_fused(BOOST_FWD_REF(F) f, util::tuple<>)
    {
        return invoke(boost::forward<F>(f));
    }
}}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/util/preprocessed/invoke_fused.hpp>
#else

#define HPX_UTIL_INVOKE_FUSED_ARG_RESULT(Z, N, D)                             \
    typename boost::add_const<BOOST_PP_CAT(Arg, N)>::type                     \
    /**/

#define HPX_UTIL_INVOKE_FUSED_ARG(Z, N, D)                                    \
    args.BOOST_PP_CAT(a, N)                                                   \
    /**/

#define HPX_UTIL_INVOKE_FUSED_FWD_ARG_RESULT(Z, N, D)                         \
    typename util::detail::add_rvalue_reference<BOOST_PP_CAT(Arg, N)>::type   \
    /**/

#define HPX_UTIL_INVOKE_FUSED_FWD_ARG(Z, N, D)                                \
    boost::forward<BOOST_PP_CAT(Arg, N)>(args.BOOST_PP_CAT(a, N))             \
    /**/

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/invoke_fused_" HPX_LIMIT_STR ".hpp")
#endif

///////////////////////////////////////////////////////////////////////////////
#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (                                                                         \
        3                                                                     \
      , (                                                                     \
            1                                                                 \
          , HPX_FUNCTION_ARGUMENT_LIMIT                                       \
          , <hpx/util/invoke_fused.hpp>                                       \
        )                                                                     \
    )                                                                         \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#undef HPX_UTIL_INVOKE_FUSED_ARG_RESULT
#undef HPX_UTIL_INVOKE_FUSED_ARG
#undef HPX_UTIL_INVOKE_FUSED_FWD_ARG_RESULT
#undef HPX_UTIL_INVOKE_FUSED_FWD_ARG

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

#else // !BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename R, typename F, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f
      , hpx::util::tuple<BOOST_PP_ENUM_PARAMS(N, Arg)> const& args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , BOOST_PP_ENUM(N, HPX_UTIL_INVOKE_FUSED_ARG, _));
    }

    template <typename R, typename F, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    BOOST_FORCEINLINE
    R
    invoke_fused_r(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<BOOST_PP_ENUM_PARAMS(N, Arg)>))) args)
    {
        return
            invoke_r<R>(boost::forward<F>(f)
              , BOOST_PP_ENUM(N, HPX_UTIL_INVOKE_FUSED_FWD_ARG, _));
    }

    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(BOOST_PP_ENUM(N, HPX_UTIL_INVOKE_FUSED_ARG_RESULT, _))
    >::type
    invoke_fused(BOOST_FWD_REF(F) f
      , hpx::util::tuple<BOOST_PP_ENUM_PARAMS(N, Arg)> const& args)
    {
        return
            invoke(boost::forward<F>(f)
              , BOOST_PP_ENUM(N, HPX_UTIL_INVOKE_FUSED_ARG, _));
    }

    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    BOOST_FORCEINLINE
    typename invoke_result_of<
        F(BOOST_PP_ENUM(N, HPX_UTIL_INVOKE_FUSED_FWD_ARG_RESULT, _))
    >::type
    invoke_fused(BOOST_FWD_REF(F) f, BOOST_RV_REF(HPX_UTIL_STRIP((
        hpx::util::tuple<BOOST_PP_ENUM_PARAMS(N, Arg)>))) args)
    {
        return
            invoke(boost::forward<F>(f)
              , BOOST_PP_ENUM(N, HPX_UTIL_INVOKE_FUSED_FWD_ARG, _));
    }
}}

#undef N

#endif

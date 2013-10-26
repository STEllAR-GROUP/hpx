//  Copyright (c) 2011-2012 Thomas Heller
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#ifndef HPX_UTIL_DETAIL_DEFINE_BIND_FUNCTION_OPERATORS_HPP
#define HPX_UTIL_DETAIL_DEFINE_BIND_FUNCTION_OPERATORS_HPP

#if defined(HPX_CREATE_PREPROCESSED_FILES)
#   include <boost/preprocessor/cat.hpp>
#   include <boost/preprocessor/iteration/iterate.hpp>
#   include <boost/preprocessor/repetition/enum.hpp>
#   include <boost/preprocessor/repetition/enum_params.hpp>
#endif

    template <typename This>
    struct result<This()>
      : bind_invoke_impl<F, BoundArgs, util::tuple<> >
    {};

    BOOST_FORCEINLINE
    typename result<bound()>::type
    operator()()
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple()
            );
    }

    template <typename This>
    struct result<This const()>
      : bind_invoke_impl<F const, BoundArgs const, util::tuple<> >
    {};

    BOOST_FORCEINLINE
    typename result<bound const()>::type
    operator()() const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple()
            );
    }

#   if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#       include <hpx/util/detail/preprocessed/define_bind_function_operators.hpp>
#   else
#       if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#           pragma wave option(preserve: 1, line: 0, output: "preprocessed/define_bind_function_operators_" HPX_LIMIT_STR ".hpp")
#       endif

        ///////////////////////////////////////////////////////////////////////
#       define BOOST_PP_ITERATION_PARAMS_1                                    \
        (                                                                     \
            3                                                                 \
          , (                                                                 \
                1                                                             \
              , HPX_FUNCTION_ARGUMENT_LIMIT                                   \
              , <hpx/util/detail/define_bind_function_operators.hpp>          \
            )                                                                 \
        )                                                                     \
        /**/
#       include BOOST_PP_ITERATE()

#       if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#           pragma wave option(output: null)
#       endif
#   endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

#else // !BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

#   define HPX_UTIL_BIND_ADD_RVALUE_REF(Z, N, D)                              \
    typename util::add_rvalue_reference<BOOST_PP_CAT(U, N)>::type             \
    /**/

    template <typename This, BOOST_PP_ENUM_PARAMS(N, typename U)>
    struct result<This(BOOST_PP_ENUM_PARAMS(N, U))>
      : bind_invoke_impl<
            F, BoundArgs
          , util::tuple<BOOST_PP_ENUM(N, HPX_UTIL_BIND_ADD_RVALUE_REF, _)>
        >
    {};

    template <BOOST_PP_ENUM_PARAMS(N, typename U)>
    BOOST_FORCEINLINE
    typename result<bound(BOOST_PP_ENUM_PARAMS(N, U))>::type
    operator()(HPX_ENUM_FWD_ARGS(N, U, u))
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(HPX_ENUM_FORWARD_ARGS(N, U, u))
            );
    }

    template <typename This, BOOST_PP_ENUM_PARAMS(N, typename U)>
    struct result<This const(BOOST_PP_ENUM_PARAMS(N, U))>
      : bind_invoke_impl<
            F const, BoundArgs const
          , util::tuple<BOOST_PP_ENUM(N, HPX_UTIL_BIND_ADD_RVALUE_REF, _)>
        >
    {};

    template <BOOST_PP_ENUM_PARAMS(N, typename U)>
    BOOST_FORCEINLINE
    typename result<bound const(BOOST_PP_ENUM_PARAMS(N, U))>::type
    operator()(HPX_ENUM_FWD_ARGS(N, U, u)) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(HPX_ENUM_FORWARD_ARGS(N, U, u))
            );
    }

#   undef HPX_UTIL_BIND_ADD_RVALUE_REF

#undef N

#endif // !BOOST_PP_IS_ITERATING

//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_UTIL_DETAIL_VTABLE_CALLABLE_VTABLE_HPP
#define HPX_UTIL_DETAIL_VTABLE_CALLABLE_VTABLE_HPP

#include <hpx/config/forceinline.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/move.hpp>

#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>

#include <typeinfo>

namespace hpx { namespace util { namespace detail
{
    template <typename Sig>
    struct callable_vtable;

    //template <typename R, typename ...Args>
    //struct callable_vtable<R(Args...)>
    //{
    //    template <typename T>
    //    BOOST_FORCEINLINE static R invoke(void** f, Args&&... args)
    //    {
    //        return util::invoke_r<R>(vtable::get<T>(f),
    //            std::forward<Args>(args)...);
    //    }
    //    typedef R (*invoke_t)(void**, Args&&...);
    //};

    template <typename R>
    struct callable_vtable<R()>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f)
        {
            return util::invoke_r<R>(vtable::get<T>(f));
        }
        typedef R (*invoke_t)(void**);
    };
}}}

#   if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#       include <hpx/util/detail/vtable/preprocessed/callable_vtable.hpp>
#   else
#       if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#           pragma wave option(preserve: 1, line: 0, output: "preprocessed/callable_vtable_" HPX_LIMIT_STR ".hpp")
#       endif

        ///////////////////////////////////////////////////////////////////////
#       define BOOST_PP_ITERATION_PARAMS_1                                    \
        (                                                                     \
            3                                                                 \
          , (                                                                 \
                1                                                             \
              , HPX_FUNCTION_ARGUMENT_LIMIT                                   \
              , <hpx/util/detail/vtable/callable_vtable.hpp>                  \
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

namespace hpx { namespace util { namespace detail
{
    template <typename R, BOOST_PP_ENUM_PARAMS(N, typename A)>
    struct callable_vtable<R(BOOST_PP_ENUM_PARAMS(N, A))>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            BOOST_PP_ENUM_BINARY_PARAMS(N, A, && a))
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                HPX_ENUM_FORWARD_ARGS(N, A, a));
        }
        typedef R (*invoke_t)(void**,
            BOOST_PP_ENUM_BINARY_PARAMS(N, A, && BOOST_PP_INTERCEPT));
    };
}}}

#undef N

#endif

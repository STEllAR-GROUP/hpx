//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_UTIL_DETAIL_EMPTY_FUNCTION_HPP
#define HPX_UTIL_DETAIL_EMPTY_FUNCTION_HPP

#include <hpx/error.hpp>
#include <hpx/util/serialize_empty_type.hpp>

#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig>
    struct empty_function; // must be trivial and empty

    template <typename R>
    struct empty_function<R()>
    {
        R operator()() const
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_function::operator()");
        }
    };
}}}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/util/detail/preprocessed/empty_function.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/empty_function_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        3                                                                       \
      , (                                                                       \
            1                                                                   \
          , HPX_FUNCTION_ARGUMENT_LIMIT                                         \
          , <hpx/util/detail/empty_function.hpp>                                \
        )                                                                       \
    )                                                                           \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

#else

#define N BOOST_PP_ITERATION()

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename R, BOOST_PP_ENUM_PARAMS(N, typename A)>
    struct empty_function<R(BOOST_PP_ENUM_PARAMS(N, A))>
    {
        R operator()(BOOST_PP_ENUM_PARAMS(N, A)) const
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_function::operator()");
        }
    };
}}}

#undef N

#endif

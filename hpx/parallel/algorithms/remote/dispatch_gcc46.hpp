//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_PARALLEL_ALGORITHM_REMOTE_DISPATCH_GCC46_JAN_08_2014_0935AM)
#define HPX_PARALLEL_ALGORITHM_REMOTE_DISPATCH_GCC46_JAN_08_2014_0935AM

#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (2, 6, "hpx/parallel/algorithms/remote/dispatch_gcc46.hpp"))          \
    /**/

#define HPX_DISPATCH_CONST_ARG(Z, N, D) BOOST_PP_CAT(D, N) const&             \
    /**/

#include BOOST_PP_ITERATE()

#undef HPX_DISPATCH_CONST_ARG

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

HPX_REGISTER_PLAIN_ACTION_TEMPLATE(
    (template <typename Algo, typename ExPolicy, typename IsSeq, typename R,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>),
    (hpx::parallel::util::remote::algorithm_invoker_action<
        Algo, ExPolicy, IsSeq, R(BOOST_PP_ENUM(N, HPX_DISPATCH_CONST_ARG, Arg))>))

#undef N

#endif

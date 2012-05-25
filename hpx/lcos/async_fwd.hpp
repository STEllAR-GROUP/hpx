//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_ASYNC_FWD_SEP_28_2011_0840AM)
#define HPX_LCOS_ASYNC_FWD_SEP_28_2011_0840AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, int Action, typename Result,
        typename Arguments, typename Derived, threads::thread_priority Priority>
    lcos::future<typename traits::promise_local_result<Result>::type, Result>
    async (
        hpx::actions::action<
            Component, Action, Result, Arguments, Derived, Priority
        > /*act*/, naming::id_type const& gid);
}

///////////////////////////////////////////////////////////////////////////////
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define HPX_FWD_ARGS(z, n, _)                                                 \
        BOOST_PP_COMMA_IF(n)                                                  \
            BOOST_FWD_REF(BOOST_PP_CAT(Arg, n)) BOOST_PP_CAT(arg, n)          \
    /**/

#define HPX_FORWARD_ARGS(z, n, _)                                             \
        BOOST_PP_COMMA_IF(n)                                                  \
            boost::forward<BOOST_PP_CAT(Arg, n)>(BOOST_PP_CAT(arg, n))        \
    /**/

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/lcos/async_fwd.hpp"))                                                \
    /**/

#include BOOST_PP_ITERATE()

#undef HPX_FWD_ARGS
#undef HPX_FORWARD_ARGS

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, int Action, typename Result,
        typename Arguments, typename Derived, threads::thread_priority Priority,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    lcos::future<typename traits::promise_local_result<Result>::type, Result>
    async (
        hpx::actions::action<
            Component, Action, Result, Arguments, Derived, Priority
        > /*act*/, naming::id_type const& gid, BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _));
}

#undef N

#endif

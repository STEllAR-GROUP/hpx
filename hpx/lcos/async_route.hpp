//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_ASYNC_SEP_28_2011_0840AM)
#define HPX_LCOS_ASYNC_SEP_28_2011_0840AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/packaged_action_route.hpp>
#include <hpx/util/move.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::result_type
        >::type,
        typename hpx::actions::extract_action<Action>::result_type
    >
    async (naming::id_type const& gid)
    {
        typedef hpx::actions::extract_action<Action>::type action_type;
        return packaged_action_route<action_type>(gid).get_future();
    }
}}

///////////////////////////////////////////////////////////////////////////////
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/lcos/async_route.hpp"))                                              \
    /**/

#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

namespace hpx { namespace lcos
{
    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::result_type
        >::type,
        typename hpx::actions::extract_action<Action>::result_type
    >
    async (naming::id_type const& gid,
        HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        typedef hpx::actions::extract_action<Action>::type action_type;
        return packaged_action_route<action_type>(gid,
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg)).get_future();
    }
}}

#undef N

#endif

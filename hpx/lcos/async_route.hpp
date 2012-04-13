//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_ASYNC_SEP_28_2011_0840AM)
#define HPX_LCOS_ASYNC_SEP_28_2011_0840AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/packaged_task_route.hpp>

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
        return packaged_task_route<action_type>(gid).get_future();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::result_type
        >::type,
        typename hpx::actions::extract_action<Action>::result_type
    >
    async_callback (
        HPX_STD_FUNCTION<void(future<typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::result_type
        >::type>)> const& data_sink, naming::id_type const& gid)
    {
        typedef hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::result_type
        >::type result_type;
        typedef packaged_task_route<action_type, result_type> future_type;

        return future_type(gid, data_sink).get_future();
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
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        typedef hpx::actions::extract_action<Action>::type action_type;
        return packaged_task_route<action_type>(gid,
            BOOST_PP_ENUM_PARAMS(N, arg)).get_future();
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::result_type
        >::type,
        typename hpx::actions::extract_action<Action>::result_type
    >
    async_callback (
        HPX_STD_FUNCTION<void(future<typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::result_type
        >::type>)> const& data_sink, naming::id_type const& gid,
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        typedef hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::result_type
        >::type result_type;
        typedef packaged_task_route<action_type, result_type> future_type;

        return future_type(gid, data_sink,
            BOOST_PP_ENUM_PARAMS(N, arg)).get_future();
    }
}}

#undef N

#endif

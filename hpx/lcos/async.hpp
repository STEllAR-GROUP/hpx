//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_ASYNC_SEP_28_2011_0840AM)
#define HPX_LCOS_ASYNC_SEP_28_2011_0840AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/lcos/future.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    future<
        typename traits::promise_local_result<
            typename Action::result_type
        >::type,
        typename Action::result_type
    >
    async (naming::id_type const& gid)
    {
        return eager_future<Action>(gid).get_future();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    future<
        typename traits::promise_local_result<
            typename Action::result_type
        >::type,
        typename Action::result_type
    >
    async_callback (
        HPX_STD_FUNCTION<void(typename traits::promise_local_result<
            typename Action::result_type
        >::type const&)> const& data_sink, naming::id_type const& gid)
    {
        typedef typename traits::promise_local_result<
            typename Action::result_type
        >::type result_type;
        typedef eager_future<Action, result_type, signalling_tag> future_type;

        return future_type(gid, data_sink).get_future();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    future<
        typename traits::promise_local_result<
            typename Action::result_type
        >::type,
        typename Action::result_type
    >
    async_callback (
        HPX_STD_FUNCTION<void(typename traits::promise_local_result<
            typename Action::result_type
        >::type const&)> const& data_sink,
        HPX_STD_FUNCTION<void(boost::exception_ptr const&)> const& error_sink,
        naming::id_type const& gid)
    {
        typedef typename traits::promise_local_result<
            typename Action::result_type
        >::type result_type;
        typedef eager_future<Action, result_type, signalling_tag> future_type;

        return future_type(gid, data_sink, error_sink).get_future();
    }
}}

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
    "hpx/lcos/async.hpp"))                                                    \
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

namespace hpx { namespace lcos
{
    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    future<
        typename traits::promise_local_result<
            typename Action::result_type
        >::type,
        typename Action::result_type
    >
    async (naming::id_type const& gid,
        BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        return eager_future<Action>(
            gid, BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _)).get_future();
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    future<
        typename traits::promise_local_result<
            typename Action::result_type
        >::type,
        typename Action::result_type
    >
    async_callback (
        HPX_STD_FUNCTION<void(typename traits::promise_local_result<
            typename Action::result_type
        >::type const&)> const& data_sink, naming::id_type const& gid,
        BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        typedef typename traits::promise_local_result<
            typename Action::result_type
        >::type result_type;
        typedef eager_future<Action, result_type, signalling_tag> future_type;

        return future_type(gid, data_sink,
            BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _)).get_future();
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    future<
        typename traits::promise_local_result<
            typename Action::result_type
        >::type,
        typename Action::result_type
    >
    async_callback (
        HPX_STD_FUNCTION<void(typename traits::promise_local_result<
            typename Action::result_type
        >::type const&)> const& data_sink,
        HPX_STD_FUNCTION<void(boost::exception_ptr const&)> const& error_sink,
        naming::id_type const& gid,
        BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        typedef typename traits::promise_local_result<
            typename Action::result_type
        >::type result_type;
        typedef eager_future<Action, result_type, signalling_tag> future_type;

        return future_type(gid, data_sink, error_sink,
            BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _)).get_future();
    }
}}

#undef N

#endif

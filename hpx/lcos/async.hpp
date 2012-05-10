//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_ASYNC_SEP_28_2011_0840AM)
#define HPX_LCOS_ASYNC_SEP_28_2011_0840AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/lcos/packaged_action.hpp>
#include <hpx/lcos/future.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::result_type
        >::type,
        typename hpx::actions::extract_action<Action>::result_type
    >
    async (BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::result_type
        >::type result_type;
        lcos::packaged_action<action_type, result_type> p;
        if (policy & launch::async)
            p.apply(gid);
        return p.get_future();
    }

    template <typename Action>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::result_type
        >::type,
        typename hpx::actions::extract_action<Action>::result_type
    >
    async (naming::id_type const& gid)
    {
        return async<Action>(launch::all, gid);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, int Action, typename Result,
        typename Arguments, typename Derived, threads::thread_priority Priority>
    lcos::future<typename traits::promise_local_result<Result>::type, Result>
    async (BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::action<
            Component, Action, Result, Arguments, Derived, Priority
        > /*act*/, naming::id_type const& gid)
    {
        lcos::packaged_action<Derived, Result> p;
        if (policy & launch::async)
            p.apply(gid);
        return p.get_future();
    }

    template <typename Component, int Action, typename Result,
        typename Arguments, typename Derived, threads::thread_priority Priority>
    lcos::future<typename traits::promise_local_result<Result>::type, Result>
    async (
        hpx::actions::action<
            Component, Action, Result, Arguments, Derived, Priority
        > /*act*/, naming::id_type const& gid)
    {
        return async<Derived>(launch::all, gid);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::result_type
        >::type,
        typename hpx::actions::extract_action<Action>::result_type
    >
    async_callback (BOOST_SCOPED_ENUM(launch) policy,
        HPX_STD_FUNCTION<void(lcos::future<typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::result_type
        >::type>)> const& data_sink, naming::id_type const& gid)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::result_type
        >::type result_type;
        typedef lcos::packaged_action<action_type, result_type>
            packaged_action_type;

        packaged_action_type p(data_sink);
        if (policy & launch::async)
            p.apply(gid);
        return p.get_future();
    }

    template <typename Action>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::result_type
        >::type,
        typename hpx::actions::extract_action<Action>::result_type
    >
    async_callback (
        HPX_STD_FUNCTION<void(lcos::future<typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::result_type
        >::type>)> const& data_sink, naming::id_type const& gid)
    {
        return async_callback<Action>(launch::all, data_sink, gid);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, int Action, typename Result,
        typename Arguments, typename Derived, threads::thread_priority Priority>
    lcos::future<typename traits::promise_local_result<Result>::type, Result>
    async_callback (BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::action<
            Component, Action, Result, Arguments, Derived, Priority
        > /*act*/,
        HPX_STD_FUNCTION<void(
            lcos::future<typename traits::promise_local_result<Result>::type>
        )> const& data_sink,
        naming::id_type const& gid)
    {
        lcos::packaged_action<Derived, Result> p(data_sink);
        if (policy & launch::async)
            p.apply(gid);
        return p.get_future();
    }

    template <typename Component, int Action, typename Result,
        typename Arguments, typename Derived, threads::thread_priority Priority>
    lcos::future<typename traits::promise_local_result<Result>::type, Result>
    async_callback (
        hpx::actions::action<
            Component, Action, Result, Arguments, Derived, Priority
        > /*act*/,
        HPX_STD_FUNCTION<void(
            lcos::future<typename traits::promise_local_result<Result>::type>
        )> const& data_sink,
        naming::id_type const& gid)
    {
        return async_callback<Derived>(launch::all, data_sink, gid);
    }
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

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::result_type
        >::type,
        typename hpx::actions::extract_action<Action>::result_type
    >
    async (BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid,
        BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::result_type
        >::type result_type;
        typedef lcos::packaged_action<action_type, result_type>
            packaged_action_type;

        packaged_action_type p;
        if (policy & launch::async)
            p.apply(gid, BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        return p.get_future();
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::result_type
        >::type,
        typename hpx::actions::extract_action<Action>::result_type
    >
    async (naming::id_type const& gid, BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        return async<Action>(launch::all, gid,
            BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, int Action, typename Result,
        typename Arguments, typename Derived, threads::thread_priority Priority,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    lcos::future<typename traits::promise_local_result<Result>::type, Result>
    async (BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::action<
            Component, Action, Result, Arguments, Derived, Priority
        > /*act*/, naming::id_type const& gid, BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        lcos::packaged_action<Derived, Result> p;
        if (policy & launch::async)
            p.apply(gid, BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        return p.get_future();
    }

    template <typename Component, int Action, typename Result,
        typename Arguments, typename Derived, threads::thread_priority Priority,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    lcos::future<typename traits::promise_local_result<Result>::type, Result>
    async (
        hpx::actions::action<
            Component, Action, Result, Arguments, Derived, Priority
        > /*act*/, naming::id_type const& gid, BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        return async<Derived>(launch::all, gid,
            BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::result_type
        >::type,
        typename hpx::actions::extract_action<Action>::result_type
    >
    async_callback (BOOST_SCOPED_ENUM(launch) policy,
        HPX_STD_FUNCTION<void(lcos::future<typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::result_type
        >::type>)> const& data_sink, naming::id_type const& gid,
        BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::result_type
        >::type result_type;
        typedef lcos::packaged_action<action_type, result_type>
            packaged_action_type;

        packaged_action_type p(data_sink);
        if (policy & launch::async)
            p.apply(gid, BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        return p.get_future();
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::result_type
        >::type,
        typename hpx::actions::extract_action<Action>::result_type
    >
    async_callback (
        HPX_STD_FUNCTION<void(lcos::future<typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::result_type
        >::type>)> const& data_sink, naming::id_type const& gid,
        BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        return async_callback<Action>(launch::all, data_sink, gid,
            BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, int Action, typename Result,
        typename Arguments, typename Derived, threads::thread_priority Priority,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    lcos::future<typename traits::promise_local_result<Result>::type, Result>
    async_callback (BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::action<
            Component, Action, Result, Arguments, Derived, Priority
        > /*act*/,
        HPX_STD_FUNCTION<void(
            lcos::future<typename traits::promise_local_result<Result>::type>
        )> const& data_sink,
        naming::id_type const& gid,
        BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        lcos::packaged_action<Derived, Result> p(data_sink);
        if (policy & launch::async)
            p.apply(gid, BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        return p.get_future();
    }

    template <typename Component, int Action, typename Result,
        typename Arguments, typename Derived, threads::thread_priority Priority,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    lcos::future<typename traits::promise_local_result<Result>::type, Result>
    async_callback (
        hpx::actions::action<
            Component, Action, Result, Arguments, Derived, Priority
        > /*act*/,
        HPX_STD_FUNCTION<void(
            lcos::future<typename traits::promise_local_result<Result>::type>
        )> const& data_sink,
        naming::id_type const& gid,
        BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        return async_callback<Derived>(launch::all, data_sink, gid,
            BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
    }
}

#undef N

#endif

//  Copyright (c) 2007-2013 Hartmut Kaiser
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
#include <hpx/lcos/async_fwd.hpp>

#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    namespace detail
    {
        BOOST_FORCEINLINE bool has_async_policy(BOOST_SCOPED_ENUM(launch) policy)
        {
            return (static_cast<int>(policy) &
                (static_cast<int>(launch::async)|static_cast<int>(launch::task))) ?
                    true : false;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::result_type
        >::type
    >
    async (BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::result_type
        >::type result_type;

        lcos::packaged_action<action_type, result_type> p;
        if (detail::has_async_policy(policy))
            p.apply(gid);
        return p.get_future();
    }

    template <typename Action>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::result_type
        >::type
    >
    async (naming::id_type const& gid)
    {
        return async<Action>(launch::all, gid);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Result,
        typename Arguments, typename Derived>
    lcos::future<typename traits::promise_local_result<Result>::type>
    async (BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /*act*/, naming::id_type const& gid)
    {
        return async<Derived>(policy, gid);
    }

    template <typename Component, typename Result,
        typename Arguments, typename Derived>
    lcos::future<typename traits::promise_local_result<Result>::type>
    async (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const & /*act*/, naming::id_type const& gid)
    {
        return async<Derived>(launch::all, gid);
    }
}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/async.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/async_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/lcos/async.hpp"))                                                    \
    /**/

#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

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
        >::type
    >
    async (BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid,
        HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::result_type
        >::type result_type;
        typedef lcos::packaged_action<action_type, result_type>
            packaged_action_type;

        packaged_action_type p;
        if (detail::has_async_policy(policy))
            p.apply(gid, HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        return p.get_future();
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::result_type
        >::type
    >
    async (naming::id_type const& gid, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return async<Action>(launch::all, gid,
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Result,
        typename Arguments, typename Derived, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    lcos::future<typename traits::promise_local_result<Result>::type>
    async (BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const & /*act*/, naming::id_type const& gid, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return async<Derived>(policy, gid,
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    template <typename Component, typename Result,
        typename Arguments, typename Derived, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    lcos::future<typename traits::promise_local_result<Result>::type>
    async (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /*act*/ const &, naming::id_type const& gid, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return async<Derived>(launch::all, gid,
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }
}

#undef N

#endif

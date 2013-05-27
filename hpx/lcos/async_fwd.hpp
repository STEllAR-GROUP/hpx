//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_ASYNC_FWD_SEP_28_2011_0840AM)
#define HPX_LCOS_ASYNC_FWD_SEP_28_2011_0840AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits.hpp>
#include <hpx/util/move.hpp>

#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    template <typename Component, typename Result,
        typename Arguments, typename Derived>
    struct action;

    // This template meta function can be used to extract the action type, no
    // matter whether it got specified directly or by passing the
    // corresponding make_action<> specialization.
    template <typename Action, typename Enable = void>
    struct extract_action
    {
        typedef typename Action::derived_type type;
        typedef typename type::result_type result_type;
        typedef typename type::remote_result_type remote_result_type;
    };

    template <typename Action>
    struct extract_action<Action, typename Action::type>
    {
        typedef typename Action::type type;
        typedef typename type::result_type result_type;
        typedef typename type::remote_result_type remote_result_type;
    };
}}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/async_fwd.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/async_fwd_" HPX_LIMIT_STR ".hpp")
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Result,
        typename Arguments, typename Derived>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const & /*act*/, naming::id_type const& gid);

    template <typename Component, typename Result,
        typename Arguments, typename Derived>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async (BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /*act*/, naming::id_type const& gid);
}

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/lcos/async_fwd.hpp"))                                                \
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
    template <typename Component, typename Result,
        typename Arguments, typename Derived, 
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const & /*act*/, naming::id_type const& gid, HPX_ENUM_FWD_ARGS(N, Arg, arg));

    template <typename Component, typename Result,
        typename Arguments, typename Derived, 
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async (BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const & /*act*/, naming::id_type const& gid, HPX_ENUM_FWD_ARGS(N, Arg, arg));
}

#undef N

#endif

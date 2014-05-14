//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_ASYNC_CONTINUE_FWD_JAN_25_2013_0828AM)
#define HPX_LCOS_ASYNC_CONTINUE_FWD_JAN_25_2013_0828AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/lcos/future.hpp>

#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename F>
    struct result_of_async_continue
        : actions::detail::remote_action_result<
            typename util::result_of<typename util::decay<F>::type(
                naming::id_type,
                typename hpx::actions::extract_action<
                    Action
                >::remote_result_type
            )>::type
        >
    {};
}}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/async_continue_fwd.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/async_continue_fwd_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (0, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/lcos/async_continue_fwd.hpp"))                                       \
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
    namespace detail
    {
        template <
            typename Action
          , typename RemoteResult
          BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)
          , typename F>
        typename boost::enable_if_c<
            util::tuple_size<typename Action::arguments_type>::value == N
          , lcos::future<
                typename traits::promise_local_result<
                    typename util::result_of_async_continue<Action, F>::type
                >::type
            >
        >::type
        async_continue_r(
            naming::id_type const& gid
          BOOST_PP_COMMA_IF(N) HPX_ENUM_FWD_ARGS(N, Arg, arg)
          , F && f);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Action
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == N
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Action, F>::type
            >::type
        >
    >::type
    async_continue(
        naming::id_type const& gid
      BOOST_PP_COMMA_IF(N) HPX_ENUM_FWD_ARGS(N, Arg, arg)
      , F && f);

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename Result, typename Arguments, typename Derived
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == N
      , lcos::future<
            typename traits::promise_local_result<
                typename util::result_of_async_continue<Derived, F>::type
            >::type
        >
    >::type
    async_continue(
        hpx::actions::action<Component, Result, Arguments, Derived> /*act*/
      , naming::id_type const& gid
      BOOST_PP_COMMA_IF(N) HPX_ENUM_FWD_ARGS(N, Arg, arg)
      , F && f);
}

#undef N

#endif

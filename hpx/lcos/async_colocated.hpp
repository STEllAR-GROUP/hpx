//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_ASYNC_COLOCATED_FEB_01_2014_0105PM)
#define HPX_LCOS_ASYNC_COLOCATED_FEB_01_2014_0105PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits.hpp>
#include <hpx/runtime/agas/request.hpp>
#include <hpx/runtime/agas/stubs/primary_namespace.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/async_fwd.hpp>
#include <hpx/lcos/async_continue_fwd.hpp>
#include <hpx/lcos/async_colocated_fwd.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/bind_action.hpp>
#include <hpx/util/functional/colocated_helpers.hpp>

#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

namespace hpx { namespace detail
{
    template <typename Tuple>
    struct async_colocated_bound_tuple;

    template <
        BOOST_PP_ENUM_PARAMS(
            HPX_TUPLE_LIMIT
          , typename T
        )
    >
    struct async_colocated_bound_tuple<
        util::tuple<
            BOOST_PP_ENUM_PARAMS(HPX_TUPLE_LIMIT, T)
        >
    >
    {
        typedef
            util::tuple<
                hpx::util::detail::bound<
                    hpx::util::functional::extract_locality
                  , hpx::util::tuple<hpx::util::detail::placeholder<2ul> >
                >
              , BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(HPX_TUPLE_LIMIT), T)
            >
            type;
    };
}}

#define HPX_REGISTER_ASYNC_COLOCATED_DECLARATION(Action, Name)                \
    HPX_UTIL_REGISTER_FUNCTION_DECLARATION(                                   \
        void (hpx::naming::id_type, hpx::agas::response)                      \
      , (hpx::util::functional::detail::async_continuation_impl<              \
            hpx::util::detail::bound_action<                                  \
                Action                                                        \
              , hpx::detail::async_colocated_bound_tuple<                     \
                    Action ::arguments_type                                   \
                >::type                                                       \
            >                                                                 \
        >)                                                                    \
      , Name                                                                  \
    );                                                                        \
/**/

#define HPX_REGISTER_ASYNC_COLOCATED(Action, Name)                            \
    HPX_UTIL_REGISTER_FUNCTION(                                               \
        void (hpx::naming::id_type, hpx::agas::response)                      \
      , (hpx::util::functional::detail::apply_continuation_impl<              \
            hpx::util::detail::bound_action<                                  \
                Action                                                        \
              , hpx::detail::apply_colocated_bound_tuple<                     \
                    Action ::arguments_type                                   \
                >::type                                                       \
            >                                                                 \
        >)                                                                    \
      , Name                                                                  \
    );                                                                        \
/**/

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/async_colocated.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/async_colocated_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (0, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/lcos/async_colocated.hpp"))                                          \
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
    template <
        typename Action
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == N
      , lcos::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::remote_result_type
            >::type>
    >::type
    async_colocated(
        naming::id_type const& gid
      BOOST_PP_COMMA_IF(N) HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        // Attach the requested action as a continuation to a resolve_async
        // call on the locality responsible for the target gid.
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);

        typedef agas::server::primary_namespace::service_action action_type;

        using util::placeholders::_2;
        return async_continue<action_type>(
            service_target, req
          , util::functional::async_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(gid), _2)
                  BOOST_PP_COMMA_IF(N) HPX_ENUM_FORWARD_ARGS(N, Arg, arg))
                ));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename Result, typename Arguments, typename Derived
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == N
      , lcos::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<Derived>::remote_result_type
            >::type>
    >::type
    async_colocated(
        hpx::actions::action<Component, Result, Arguments, Derived> /*act*/
      , naming::id_type const& gid
      BOOST_PP_COMMA_IF(N) HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return async_colocated<Derived>(
            gid
          BOOST_PP_COMMA_IF(N) HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }
}

#undef N

#endif

//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_APPLIER_APPLY_COLOCATED_FEB_04_2014_0755PM)
#define HPX_RUNTIME_APPLIER_APPLY_COLOCATED_FEB_04_2014_0755PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/agas/request.hpp>
#include <hpx/runtime/agas/stubs/primary_namespace.hpp>
#include <hpx/runtime/applier/apply_continue.hpp>
#include <hpx/runtime/applier/register_apply_colocated.hpp>
#include <hpx/util/functional/colocated_helpers.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/bind_action.hpp>

#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/runtime/applier/preprocessed/apply_colocated.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/apply_colocated_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (0, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/applier/apply_colocated.hpp"))                               \
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
    template <typename Action
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == N
      , bool
    >::type
    apply_colocated(
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
        return apply_continue<action_type>(
            service_target, req
          , util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  BOOST_PP_COMMA_IF(N) HPX_ENUM_FORWARD_ARGS(N, Arg, arg))
                ));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename Result, typename Arguments, typename Derived
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == N
      , bool
    >::type
    apply_colocated(
        hpx::actions::action<Component, Result, Arguments, Derived> /*act*/
      , naming::id_type const& gid
      BOOST_PP_COMMA_IF(N) HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return apply_colocated<Derived>(
            gid
          BOOST_PP_COMMA_IF(N) HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }
}

#undef N

#endif

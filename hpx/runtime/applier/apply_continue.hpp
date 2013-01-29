//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_APPLIER_APPLY_CONTINUE_JAN_27_2013_0726PM)
#define HPX_RUNTIME_APPLIER_APPLY_CONTINUE_JAN_27_2013_0726PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/applier/apply.hpp>

#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<
            typename Action::arguments_type>::value == 0>
    >::type
    apply_continue(naming::id_type const& gid, BOOST_FWD_REF(F) f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type remote_result_type;

        apply<Action>(
            new hpx::actions::typed_continuation<remote_result_type>(
                boost::forward<F>(f))
          , gid);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<Arguments>::value == 0>
    >::type
    apply_continue(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /*act*/, naming::id_type const& gid, BOOST_FWD_REF(F) f)
    {
        apply_continue<Derived>(gid, boost::forward<F>(f));
    }
}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/runtime/applier/preprocessed/apply_continue.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/apply_continue_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/applier/apply_continue.hpp"))                                \
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
    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg),
        typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<
            typename Action::arguments_type>::value == N>
    >::type
    apply_continue(naming::id_type const& gid, HPX_ENUM_FWD_ARGS(N, Arg, arg),
        BOOST_FWD_REF(F) f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type remote_result_type;

        apply<Action>(
            new hpx::actions::typed_continuation<remote_result_type>(
                boost::forward<F>(f))
          , gid
          , HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Result, typename Arguments,
        typename Derived, BOOST_PP_ENUM_PARAMS(N, typename Arg), typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<Arguments>::value == N>
    >::type
    apply_continue(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /*act*/, naming::id_type const& gid, HPX_ENUM_FWD_ARGS(N, Arg, arg),
        BOOST_FWD_REF(F) f)
    {
        return apply_continue<Derived>(gid, HPX_ENUM_FORWARD_ARGS(N, Arg, arg),
            boost::forward<F>(f));
    }
}

#undef N

#endif

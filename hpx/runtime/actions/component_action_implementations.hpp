//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_ACTIONS_ACTION_IMPLEMENTATIONS_MAY_20_2008_1104AM)
#define HPX_RUNTIME_ACTIONS_ACTION_IMPLEMENTATIONS_MAY_20_2008_1104AM

#include <hpx/config/forceinline.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

// now generate the rest, which is platform independent
#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/runtime/actions/preprocessed/component_action_implementations.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/component_action_implementations_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/actions/component_action_implementations.hpp"))              \
    /**/

#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

///////////////////////////////////////////////////////////////////////////////
// bring in all other arities for actions
#include <hpx/runtime/actions/preprocessed/component_const_action_implementations.hpp>
#include <hpx/runtime/actions/preprocessed/component_non_const_action_implementations.hpp>

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename F, F funcptr, typename Derived>
    class BOOST_PP_CAT(base_result_action, N);

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, F funcptr, typename Derived = detail::this_type>
    struct BOOST_PP_CAT(result_action, N);

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, F funcptr, typename Derived = detail::this_type>
    struct BOOST_PP_CAT(direct_result_action, N);

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, F funcptr, typename Derived>
    class BOOST_PP_CAT(base_action, N);

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, F funcptr, typename Derived = detail::this_type>
    struct BOOST_PP_CAT(action, N);

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, F funcptr, typename Derived = detail::this_type>
    struct BOOST_PP_CAT(direct_action, N);
}}

#undef N

#endif


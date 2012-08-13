//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file component_action_enum.hpp

#if !defined(HPX_RUNTIME_ACTIONS_COMPONENT_ACTION_ENUM_MAR_26_2008_1054AM)
#define HPX_RUNTIME_ACTIONS_COMPONENT_ACTION_ENUM_MAR_26_2008_1054AM

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    /// \cond NOINTERNAL

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/runtime/actions/preprocessed/component_action_enum.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/component_action_enum_" HPX_LIMIT_STR ".hpp")
#endif

    ///////////////////////////////////////////////////////////////////////////
#define HPX_FUNCTION_ARG_ENUM(z, n, data)                                     \
        BOOST_PP_CAT(component_action_arg, BOOST_PP_INC(n)) =                 \
            component_action_base + BOOST_PP_INC(n),                          \
    /**/
#define HPX_FUNCTION_RETARG_ENUM(z, n, data)                                  \
        BOOST_PP_CAT(component_result_action_arg, BOOST_PP_INC(n)) =          \
            component_result_action_base + BOOST_PP_INC(n),                   \
    /**/

    enum component_action
    {
        /// remotely callable member function identifiers
        component_action_base = 1000,
        component_action_arg0 = component_action_base + 0,
        BOOST_PP_REPEAT(HPX_ACTION_ARGUMENT_LIMIT, HPX_FUNCTION_ARG_ENUM, _)

        /// remotely callable member function identifiers with result
        component_result_action_base = 2000,
        BOOST_PP_REPEAT(HPX_ACTION_ARGUMENT_LIMIT, HPX_FUNCTION_RETARG_ENUM, _)
        component_result_action_arg0 = component_result_action_base + 0
    };

#undef HPX_FUNCTION_RETARG_ENUM
#undef HPX_FUNCTION_ARG_ENUM

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
}}

#endif

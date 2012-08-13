//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file plain_action_enum.hpp

#if !defined(HPX_RUNTIME_ACTIONS_PLAIN_ACTION_ENUM_NOV_14_2008_0706PM)
#define HPX_RUNTIME_ACTIONS_PLAIN_ACTION_ENUM_NOV_14_2008_0706PM

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    /// \cond NOINTERNAL

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/runtime/actions/preprocessed/plain_action_enum.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/plain_action_enum_" HPX_LIMIT_STR ".hpp")
#endif

    ///////////////////////////////////////////////////////////////////////////
#define HPX_FUNCTION_ARG_ENUM(z, n, data)                                     \
        BOOST_PP_CAT(function_action_arg, BOOST_PP_INC(n)) =                  \
            function_action_base + BOOST_PP_INC(n),                           \
    /**/
#define HPX_FUNCTION_RETARG_ENUM(z, n, data)                                  \
        BOOST_PP_CAT(function_result_action_arg, BOOST_PP_INC(n)) =           \
            function_result_action_base + BOOST_PP_INC(n),                    \
    /**/

    enum function_action
    {
        /// plain (free) remotely callable function identifiers
        function_action_base = 3000,
        function_action_arg0 = function_action_base + 0,
        BOOST_PP_REPEAT(HPX_ACTION_ARGUMENT_LIMIT, HPX_FUNCTION_ARG_ENUM, _)

        /// plain (free) remotely callable function identifiers with result
        function_result_action_base = 4000,
        BOOST_PP_REPEAT(HPX_ACTION_ARGUMENT_LIMIT, HPX_FUNCTION_RETARG_ENUM, _)
        function_result_action_arg0 = function_result_action_base + 0
    };

#undef HPX_FUNCTION_RETARG_ENUM
#undef HPX_FUNCTION_ARG_ENUM

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
}}

#endif

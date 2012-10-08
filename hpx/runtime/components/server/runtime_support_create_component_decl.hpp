//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_SUPPORT_CREATE_COMPONENT_DECL_OCT_08_2012_1001AM)
#define HPX_RUNTIME_SUPPORT_CREATE_COMPONENT_DECL_OCT_08_2012_1001AM

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/runtime/components/server/preprocessed/runtime_support_create_component_decl.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/runtime_support_create_component_decl_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
     "hpx/runtime/components/server/runtime_support_create_component_decl.hpp"))\
/**/

#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif // HPX_RUNTIME_SUPPORT_CREATE_COMPONENT_DECL_OCT_08_2012_1001AM

#else

#define N BOOST_PP_ITERATION()

        template <typename Component, BOOST_PP_ENUM_PARAMS(N, typename A)>
        naming::gid_type BOOST_PP_CAT(create_component, N)(
            BOOST_PP_ENUM_BINARY_PARAMS(N, A, a));

#undef N

#endif


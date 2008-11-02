//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_COMPONENTS_GENERIC_COMPONENT_EVAL_OCT_13_2008_0847AM)
#define HPX_COMPONENTS_GENERIC_COMPONENT_EVAL_OCT_13_2008_0847AM

#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/components/generic_component_eval.hpp"))                     \
    /**/
    
#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    result_type eval(threads::thread_self& self, 
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        return this->base_type::eval(self, this->gid_, BOOST_PP_ENUM_PARAMS(N, arg));
    }

#undef N

#endif

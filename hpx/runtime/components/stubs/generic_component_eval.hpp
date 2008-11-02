//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_COMPONENTS_STUBS_GENERIC_COMPONENT_IMPLEMENTTAION_OCT_12_2008_0941PM)
#define HPX_COMPONENTS_STUBS_GENERIC_COMPONENT_IMPLEMENTTAION_OCT_12_2008_0941PM

#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/components/stubs/generic_component_eval.hpp"))               \
    /**/
    
#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    static void 
    eval(threads::thread_self& self, applier::applier& appl, 
        naming::id_type const& targetgid, 
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        return detail::eval<action_type, result_type>::call(self, appl, 
            targetgid, params_type(BOOST_PP_ENUM_PARAMS(N, arg)));
    }

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    void eval(threads::thread_self& self, naming::id_type const& targetgid, 
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        eval(self, this->appl_, targetgid, BOOST_PP_ENUM_PARAMS(N, arg));
    }

#undef N

#endif

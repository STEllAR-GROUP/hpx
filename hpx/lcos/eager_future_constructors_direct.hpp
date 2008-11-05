//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_EAGER_FUTURE_CONSTRUCTORS_DIRECT_JUL_01_2008_0116PM)
#define HPX_LCOS_EAGER_FUTURE_CONSTRUCTORS_DIRECT_JUL_01_2008_0116PM

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (2, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/lcos/eager_future_constructors_direct.hpp"))                         \
    /**/
    
#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    void apply(applier::applier& appl, naming::id_type const& gid, 
            BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        naming::address addr;
        if (appl.address_is_local(gid, addr)) {
            // local, direct execution
            BOOST_ASSERT(components::types_are_compatible(
                addr.type_, Action::get_static_component_type()));
            (*this->impl_)->set_data(0, Action::execute_function(
                appl, addr.address_, BOOST_PP_ENUM_PARAMS(N, arg)));
        }
        else {
            // remote execution
            appl.apply_c<Action>(addr, this->get_gid(appl), gid, 
                BOOST_PP_ENUM_PARAMS(N, arg));
        }
    }

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    eager_future(applier::applier& appl, naming::id_type const& gid, 
            BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        apply(appl, gid, BOOST_PP_ENUM_PARAMS(N, arg));
    }

#undef N

#endif

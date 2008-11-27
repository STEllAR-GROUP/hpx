//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_LAZY_FUTURE_GET_RESULTS_DIRCET_SEP_17_2008_0126PM)
#define HPX_LCOS_LAZY_FUTURE_GET_RESULTS_DIRCET_SEP_17_2008_0126PM

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (2, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/lcos/lazy_future_get_results_direct.hpp"))                           \
    /**/
    
#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    Result get(naming::id_type const& gid, 
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        // Determine whether the gid is local or remote
        naming::address addr;
        if (hpx::applier::get_applier().address_is_local(gid, addr)) {
            // local, direct execution
            BOOST_ASSERT(components::types_are_compatible(addr.type_, 
                components::get_component_type<typename Action::component_type>()));
            return Action::execute_function(addr, BOOST_PP_ENUM_PARAMS(N, arg));
        }

        // initialize the remote operation
        hpx::applier::apply_c<Action>(addr, this->get_gid(), gid, 
            BOOST_PP_ENUM_PARAMS(N, arg));

        // wait for the result (yield control)
        return (*this->impl_)->get(0);
    }

#undef N

#endif

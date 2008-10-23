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
    Result get(threads::thread_self& self,
        applier::applier& appl, naming::id_type const& gid, 
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        // Determine whether the gid is local or remote
        naming::address addr;
        if (appl.address_is_local(gid, addr)) {
            // local, direct execution
            return Action::execute_function(appl, addr,
                BOOST_PP_ENUM_PARAMS(N, arg));
        }

        // initialize the remote operation
        appl.apply_c<Action>(addr, this->get_gid(appl), gid, 
            BOOST_PP_ENUM_PARAMS(N, arg));

        // wait for the result (yield control)
        return (*this->impl_)->get(self, 0);
    }

#undef N

#endif

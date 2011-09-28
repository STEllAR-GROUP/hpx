//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_ASYNC_SEP_28_2011_0840AM)
#define HPX_LCOS_ASYNC_SEP_28_2011_0840AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/lcos/eager_future.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos 
{
    template <typename Action>
    promise<
        typename traits::promise_local_result<
            typename Action::result_type
        >::type> 
    async (naming::id_type const& gid)
    {
        return eager_future<Action>(gid);
    }
}}

///////////////////////////////////////////////////////////////////////////////
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/lcos/async.hpp"))                                                    \
    /**/

#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

namespace hpx { namespace lcos 
{
    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    promise<
        typename traits::promise_local_result<
            typename Action::result_type
        >::type> 
    async (naming::id_type const& gid, 
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        return eager_future<Action>(gid, BOOST_PP_ENUM_PARAMS(N, arg));
    }
}}

#undef N

#endif

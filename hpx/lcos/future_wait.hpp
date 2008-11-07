//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_FUTURE_WAIT_OCT_23_2008_1140AM)
#define HPX_LCOS_FUTURE_WAIT_OCT_23_2008_1140AM

#include <hpx/hpx_fwd.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components 
{
    /// The one argument version is special in the sense that it returns the 
    /// expected value directly (without wrapping it into a tuple).
    template <typename T1>
    inline T1
    wait (threads::thread_self& self, lcos::future_value<T1>& f1)
    {
        return f1.get(self);
    }

    template <typename T1, typename T2>
    inline boost::tuple<T1, T2>
    wait (threads::thread_self& self, lcos::future_value<T1>& f1, 
        lcos::future_value<T2>& f2)
    {
        return boost::make_tuple(f1.get(self), f2.get(self));
    }

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (3, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/lcos/future_wait.hpp"))                                              \
    /**/

#include BOOST_PP_ITERATE()

}}

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()
#define HPX_FUTURE_WAIT_ARGUMENT(z, n, data) BOOST_PP_COMMA_IF(n)             \
        lcos::future_value<BOOST_PP_CAT(T, n)>& BOOST_PP_CAT(f, n)            \
    /**/
#define HPX_FUTURE_TUPLE_ARGUMENT(z, n, data) BOOST_PP_COMMA_IF(n)            \
        BOOST_PP_CAT(f, n).get(self)                                          \
    /**/

    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    inline boost::tuple<BOOST_PP_ENUM_PARAMS(N, T)>
    wait (threads::thread_self& self, 
        BOOST_PP_REPEAT(N, HPX_FUTURE_WAIT_ARGUMENT, _))
    {
        return boost::make_tuple(BOOST_PP_REPEAT(N, HPX_FUTURE_TUPLE_ARGUMENT, _));
    }

#undef HPX_FUTURE_WAIT_ARGUMENT
#undef HPX_FUTURE_TUPLE_ARGUMENT
#undef N

#endif

//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_FUTURE_WAIT_OCT_23_2008_1140AM)
#define HPX_LCOS_FUTURE_WAIT_OCT_23_2008_1140AM

#include <hpx/hpx_fwd.hpp>

#include <vector>

#include <boost/foreach.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    /// The one argument version is special in the sense that it returns the
    /// expected value directly (without wrapping it into a tuple).
    template <typename T1, typename TR1>
    inline T1
    wait (lcos::future<T1, TR1> const& f1)
    {
        return f1.get();
    }

    inline void
    wait (lcos::future<void> const& f1)
    {
        f1.get();
    }

    template <typename T1, typename T2, typename TR1, typename TR2>
    inline boost::tuple<T1, T2>
    wait (lcos::future<T1, TR1> const& f1, lcos::future<T2, TR2> const& f2)
    {
        return boost::make_tuple(f1.get(), f2.get());
    }

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (3, HPX_WAIT_ARGUMENT_LIMIT,                                          \
    "hpx/lcos/future_wait.hpp"))                                              \
    /**/

#include BOOST_PP_ITERATE()

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename TR>
    inline void
    wait (std::vector<lcos::future<T, TR> > const& v, std::vector<TR>& r)
    {
        r.reserve(v.size());

        typedef lcos::future<T, TR> value_type;
        BOOST_FOREACH(value_type const& f, v)
            r.push_back(f.get());
    }

    template <typename T, typename TR>
    inline void
    wait (std::vector<lcos::future<T, TR> > const& v)
    {
        typedef lcos::future<T, TR> value_type;
        BOOST_FOREACH(value_type const& f, v)
            f.get();
    }

    inline void
    wait (std::vector<lcos::future<void> > const& v)
    {
        typedef lcos::future<void> value_type;
        BOOST_FOREACH(value_type const& f, v)
            f.get();
    }
}}

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

#define HPX_FUTURE_WAIT_ARGUMENT(z, n, data) BOOST_PP_COMMA_IF(n)             \
        lcos::future<BOOST_PP_CAT(T, n), BOOST_PP_CAT(TR, n)> const&          \
            BOOST_PP_CAT(f, n)                                                \
    /**/
#define HPX_FUTURE_TUPLE_ARGUMENT(z, n, data) BOOST_PP_COMMA_IF(n)            \
        BOOST_PP_CAT(f, n).get()                                              \
    /**/

    template <
        BOOST_PP_ENUM_PARAMS(N, typename T),
        BOOST_PP_ENUM_PARAMS(N, typename TR)>
    inline boost::tuple<BOOST_PP_ENUM_PARAMS(N, T)>
    wait (BOOST_PP_REPEAT(N, HPX_FUTURE_WAIT_ARGUMENT, _))
    {
        return boost::make_tuple(BOOST_PP_REPEAT(N, HPX_FUTURE_TUPLE_ARGUMENT, _));
    }

#undef HPX_FUTURE_WAIT_ARGUMENT
#undef HPX_FUTURE_TUPLE_ARGUMENT
#undef N

#endif

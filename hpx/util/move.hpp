
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_MOVE_HPP
#define HPX_UTIL_MOVE_HPP

#ifdef HPX_HAVE_CXX11_RVALUE_REFERENCES
#define BOOST_MOVE_USE_STANDARD_MOVE
#endif

#include <boost/move/move.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/tuple/rem.hpp>
#include <boost/preprocessor/tuple/elem.hpp>

///////////////////////////////////////////////////////////////////////////////
#define HPX_FWD_ARGS(z, n, d)                                                 \
    BOOST_FWD_REF(BOOST_PP_CAT(BOOST_PP_TUPLE_ELEM(2, 0, d), n))              \
    BOOST_PP_CAT(BOOST_PP_TUPLE_ELEM(2, 1, d), n)                             \
    /**/
#define HPX_FORWARD_ARGS(z, n, d)                                             \
    boost::forward<BOOST_PP_CAT(BOOST_PP_TUPLE_ELEM(2, 0, d), n)>(            \
        BOOST_PP_CAT(BOOST_PP_TUPLE_ELEM(2, 1, d), n)                         \
    )                                                                         \
    /**/
#define HPX_MOVE_ARGS(z, n, d)                                                \
    boost::move(BOOST_PP_CAT(d, n))                                           \
    /**/

#define HPX_ENUM_FWD_ARGS(N, A, B)                                            \
    BOOST_PP_ENUM(N, HPX_FWD_ARGS, (A, B))                                    \
    /**/
#define HPX_ENUM_FORWARD_ARGS(N, A, B)                                        \
    BOOST_PP_ENUM(N, HPX_FORWARD_ARGS, (A, B))                                \
    /**/

#define HPX_ENUM_MOVE_ARGS(N, A)                                              \
    BOOST_PP_ENUM(N, HPX_MOVE_ARGS, A)                                        \
    /**/

#endif

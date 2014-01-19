//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_MOVE_HPP
#define HPX_UTIL_MOVE_HPP

#include <hpx/config.hpp>

#include <hpx/util/decay.hpp>

#if defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
#error HPX needs rvalue reference support
#endif

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/tuple/rem.hpp>
#include <boost/preprocessor/tuple/elem.hpp>

#include <utility>

///////////////////////////////////////////////////////////////////////////////
#define HPX_FORWARD_ARGS_(z, n, d)                                            \
    std::forward<BOOST_PP_CAT(BOOST_PP_TUPLE_ELEM(2, 0, d), n)>(              \
        BOOST_PP_CAT(BOOST_PP_TUPLE_ELEM(2, 1, d), n)                         \
    )                                                                         \
    /**/
#define HPX_MOVE_ARGS_(z, n, d)                                               \
    std::move(BOOST_PP_CAT(d, n))                                             \
    /**/

#define HPX_ENUM_FWD_ARGS(N, A, B)                                            \
    BOOST_PP_ENUM_BINARY_PARAMS(N, A, && B)                                   \
    /**/

#define HPX_ENUM_FORWARD_ARGS(N, A, B)                                        \
    BOOST_PP_ENUM(N, HPX_FORWARD_ARGS_, (A, B))                               \
    /**/
#define HPX_ENUM_MOVE_ARGS(N, A)                                              \
    BOOST_PP_ENUM(N, HPX_MOVE_ARGS_, A)                                       \
    /**/

#if defined(BOOST_NO_CXX11_DELETED_FUNCTIONS)
#define HPX_MOVABLE_BUT_NOT_COPYABLE(TYPE)                                    \
    private:                                                                  \
        TYPE(TYPE const &);                                                   \
        TYPE& operator=(TYPE const &);                                        \
/**/
#else
#define HPX_MOVABLE_BUT_NOT_COPYABLE(TYPE)                                    \
    public:                                                                   \
        TYPE(TYPE const &) = delete;                                          \
        TYPE& operator=(TYPE const &) = delete;                               \
    private:                                                                  \
/**/
#endif

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    typename decay<T>::type decay_copy(T&& v)
    {
        return std::forward<T>(v);
    }
}}}

#endif


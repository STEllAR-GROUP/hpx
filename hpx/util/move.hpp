//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_MOVE_HPP
#define HPX_UTIL_MOVE_HPP

#include <hpx/config.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/tuple/rem.hpp>
#include <boost/preprocessor/tuple/elem.hpp>

#include <boost/move/move.hpp>

#include <boost/type_traits/add_const.hpp>

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

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct make_temporary
    {
        template <typename U>
        static BOOST_RV_REF(T) call(U& u)
        {
            return boost::move(u);
        }
    };

    template <typename T>
    struct make_temporary<T&>
    {
        static T call(T& u)
        {
            return u;
        }
    };
    template <typename T>
    struct make_temporary<T const&>
    {
        static T call(T const& u)
        {
            return u;
        }
    };

    template <typename T>
    struct make_temporary<BOOST_RV_REF(T)>
    {
        template <typename U>
        static BOOST_RV_REF(T) call(U& u)
        {
            return boost::move(u);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // See class.copy [12.8]/5
    template <typename T, typename U = typename boost::add_const<T>::type>
    struct copy_construct
    {
        template <typename A>
        static T call(BOOST_FWD_REF(A) u)
        {
            return boost::forward<A>(u);
        }
    };

    template <typename T, typename U>
    struct copy_construct<T&, U&>
    {
        static T& call(U& u)
        {
            return u;
        }
    };

    template <typename T, typename U>
    struct copy_construct<BOOST_RV_REF(T), BOOST_RV_REF(U)>
    {
        static BOOST_RV_REF(T) call(U& u)
        {
            return boost::move(u);
        }
    };
}}}

#endif


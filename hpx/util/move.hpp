//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_MOVE_HPP
#define HPX_UTIL_MOVE_HPP

#include <hpx/util/detail/remove_reference.hpp>

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
#define HPX_MOVE_IF_NO_REF_ARGS(z, n, d)                                      \
    hpx::util::detail::move_if_no_ref<                                       \
        BOOST_PP_CAT(BOOST_PP_TUPLE_ELEM(2, 0, d), n)>                        \
            ::call(BOOST_PP_CAT(BOOST_PP_TUPLE_ELEM(2, 1, d), n))             \
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
#define HPX_ENUM_MOVE_IF_NO_REF_ARGS(N, A, B)                                 \
    BOOST_PP_ENUM(N, HPX_MOVE_IF_NO_REF_ARGS, (A, B))                         \
    /**/


namespace hpx { namespace util { namespace detail
{
#if !defined(BOOST_NO_RVALUE_REFERENCES)
    template <typename T, bool IsRvalueRef =
        std::is_rvalue_reference<T>::type::value>
#else
    template <typename T, bool IsRvalueRef = false>
#endif
    struct move_if_no_ref
    {
        template <typename A>
        static T call(BOOST_FWD_REF(A) t)
        {
            return boost::move(t);
        }
    };

    template <typename T>
    struct move_if_no_ref<T &, false>
    {
        template <typename A>
        static T & call(BOOST_FWD_REF(A) t)
        {
            return t;
        }
    };

    template <typename T>
    struct move_if_no_ref<T const &, false>
    {
        template <typename A>
        static T const & call(BOOST_FWD_REF(A) t)
        {
            return t;
        }
    };

    template <typename T>
    struct move_if_no_ref<T, true>
    {
        template <typename A>
        static BOOST_RV_REF(
            typename hpx::util::detail::remove_reference<T>::type)
        call(BOOST_FWD_REF(A) t)
        {
            return boost::move(t);
        }
    };
}}}

#endif


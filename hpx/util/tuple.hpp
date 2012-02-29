//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_UTIL_TUPLE_HPP
#define HPX_UTIL_TUPLE_HPP

#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/fusion/adapted/struct/adapt_struct.hpp>

#define M0(Z, N, D)                                                             \
    typedef BOOST_PP_CAT(A, N) BOOST_PP_CAT(member_type, N);                    \
    BOOST_PP_CAT(A, N) BOOST_PP_CAT(a, N);                                      \
/**/
#define M1(Z, N, D)                                                             \
    (BOOST_PP_CAT(A, N))                                                        \
/**/
#define M2(Z, N, D)                                                             \
    (BOOST_PP_CAT(T, N))                                                        \
/**/
#define M3(Z, N, D)                                                             \
    (BOOST_PP_CAT(A, N), BOOST_PP_CAT(a, N))                                    \
/**/

namespace hpx { namespace util
{
    template <typename Dummy = void>
    struct tuple0
    {
        typedef boost::mpl::int_<0> size_type;
        static const int size_value = 0;
    };
}}


#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        3                                                                       \
      , (                                                                       \
            1                                                                   \
          , HPX_FUNCTION_LIMIT                                                  \
          , <hpx/util/tuple.hpp>                                                \
        )                                                                       \
    )                                                                           \
/**/
#include BOOST_PP_ITERATE()

#undef M0
#undef M1
#undef M2
#undef M3

#endif

#else // !BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

#define HPX_UTIL_TUPLE_NAME BOOST_PP_CAT(tuple, N)

#define HPX_UTIL_TUPLE_FWD_REF_TYPES(Z, N, D)                                   \
    typename BOOST_FWD_REF(BOOST_PP_CAT(Arg, N))                                \

#define HPX_UTIL_TUPLE_FWD_REF_PARAMS(Z, N, D)                                  \
    BOOST_FWD_REF(BOOST_PP_CAT(Arg, N)) BOOST_PP_CAT(arg, N)                    \

#define HPX_UTIL_TUPLE_INIT_MEMBER(Z, N, D)                                     \
    BOOST_PP_CAT(a, N)(boost::forward<BOOST_PP_CAT(Arg, N)>(BOOST_PP_CAT(arg, N))) \

#define HPX_UTIL_TUPLE_INIT_COPY_MEMBER(Z, N, D)                                \
    BOOST_PP_CAT(a, N)(BOOST_PP_CAT(other.a, N))                                \

#define HPX_UTIL_TUPLE_INIT_MOVE_MEMBER(Z, N, D)                                \
    BOOST_PP_CAT(a, N)(boost::move(BOOST_PP_CAT(other.a, N)))                   \

#define HPX_UTIL_TUPLE_ASSIGN_COPY_MEMBER(Z, N, D)                              \
    BOOST_PP_CAT(a, N) = BOOST_PP_CAT(other.a, N);                              \

#define HPX_UTIL_TUPLE_ASSIGN_MOVE_MEMBER(Z, N, D)                              \
    BOOST_PP_CAT(a, N) = boost::move(BOOST_PP_CAT(other.a, N));                 \

namespace hpx { namespace util
{
    template <BOOST_PP_ENUM_PARAMS(N, typename A)>
    struct HPX_UTIL_TUPLE_NAME
    {
        BOOST_PP_REPEAT(N, M0, _)

        HPX_UTIL_TUPLE_NAME() {}

        HPX_UTIL_TUPLE_NAME(HPX_UTIL_TUPLE_NAME const & other)
          : BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_INIT_COPY_MEMBER, _)
        {}

        HPX_UTIL_TUPLE_NAME(BOOST_RV_REF(HPX_UTIL_TUPLE_NAME) other)
          : BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_INIT_MOVE_MEMBER, _)
        {}

        HPX_UTIL_TUPLE_NAME & operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_TUPLE_NAME) other)
        {
            BOOST_PP_REPEAT(N, HPX_UTIL_TUPLE_ASSIGN_COPY_MEMBER, _)
            return *this;
        }

        HPX_UTIL_TUPLE_NAME & operator=(BOOST_RV_REF(HPX_UTIL_TUPLE_NAME) other)
        {
            BOOST_PP_REPEAT(N, HPX_UTIL_TUPLE_ASSIGN_MOVE_MEMBER, _)
            return *this;
        }

        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        HPX_UTIL_TUPLE_NAME(BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_FWD_REF_PARAMS, _))
          : BOOST_PP_ENUM(N, HPX_UTIL_TUPLE_INIT_MEMBER, _)
        {}

        typedef boost::mpl::int_<N> size_type;
        static const int size_value = N;

    private:
        BOOST_COPYABLE_AND_MOVABLE(HPX_UTIL_TUPLE_NAME);

    };
}}

BOOST_FUSION_ADAPT_TPL_STRUCT(
    BOOST_PP_REPEAT(N, M1, _)
  , (BOOST_PP_CAT(hpx::util::tuple, N))BOOST_PP_REPEAT(N, M1, _)
  , BOOST_PP_REPEAT(N, M3, _)
)

#undef N

#endif

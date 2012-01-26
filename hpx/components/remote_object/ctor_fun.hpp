//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_COMPONENTS_REMOTE_OBJECT_CTOR_FUN_HPP
#define HPX_COMPONENTS_REMOTE_OBJECT_CTOR_FUN_HPP

#include <boost/move/move.hpp>
#include <boost/preprocessor/repetition/enum_params_with_a_default.hpp>
#include <boost/preprocessor/repetition/enum.hpp>

#ifndef HPX_FUNCTION_LIMIT
#define HPX_FUNCTION_LIMIT 10
#endif

namespace hpx { namespace components { namespace remote_object
{
    // helper functor to construct objects remotely
    template <
        typename T
      , BOOST_PP_ENUM_PARAMS_WITH_A_DEFAULT(HPX_FUNCTION_LIMIT, typename A, void)
      , typename Enable = void
    >
    struct ctor_fun;

    template <typename T>
    struct ctor_fun<T>
    {
        typedef void result_type;

        void operator()(void ** p) const
        {
            T * t = new T();
            *p = t;
        }

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };

#define HPX_REMOTE_OBJECT_M0(Z, N, D)                                           \
    BOOST_PP_CAT(a, N)(BOOST_PP_CAT(a, N))                                      \
/**/

#define HPX_REMOTE_OBJECT_MOVE_ARG(Z, N, D)                                     \
    BOOST_PP_CAT(a, N)(boost::move(BOOST_PP_CAT(a, N)))                         \
/**/

#define HPX_REMOTE_OBJECT_RV(Z, N, D)                                           \
    BOOST_RV_REF(BOOST_PP_CAT(A, N)) BOOST_PP_CAT(a, N)                         \
/**/

#define HPX_REMOTE_OBJECT_COPY(Z, N, D)                                         \
    BOOST_PP_CAT(a, N) = rhs.BOOST_PP_CAT(a, N);                                \
/**/

#define HPX_REMOTE_OBJECT_MOVE(Z, N, D)                                         \
    BOOST_PP_CAT(a, N) = boost::move(rhs.BOOST_PP_CAT(a, N));                   \
/**/

#define HPX_REMOTE_OBJECT_M1(Z, N, D)                                           \
    ar & BOOST_PP_CAT(a, N);                                                    \
/**/

#define HPX_REMOTE_OBJECT_M2(Z, N, D)                                           \
    BOOST_PP_CAT(A, N) BOOST_PP_CAT(a, N);                                      \
/**/

#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        3                                                                       \
      , (                                                                       \
            1                                                                   \
          , HPX_FUNCTION_LIMIT                                                  \
          , <hpx/components/remote_object/ctor_fun.hpp>                         \
        )                                                                       \
    )                                                                           \
/**/
#include BOOST_PP_ITERATE()

#undef HPX_REMOTE_OBJECT_M0
#undef HPX_REMOTE_OBJECT_M1
#undef HPX_REMOTE_OBJECT_M2
#undef HPX_REMOTE_OBJECT_MOVE_ARG
#undef HPX_REMOTE_OBJECT_RV
#undef HPX_REMOTE_OBJECT_MOVE
#undef HPX_REMOTE_OBJECT_COPY

}}}

#endif

#else

#define N BOOST_PP_ITERATION()

    template <typename T, BOOST_PP_ENUM_PARAMS(N, typename A)>
    struct ctor_fun<T, BOOST_PP_ENUM_PARAMS(N, A)>
    {
        typedef void result_type;

        ctor_fun() {}
        ctor_fun(BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a))
            : BOOST_PP_ENUM(N, HPX_REMOTE_OBJECT_M0, _)
        {}

        ctor_fun(BOOST_PP_ENUM(N, HPX_REMOTE_OBJECT_RV, _))
            : BOOST_PP_ENUM(N, HPX_REMOTE_OBJECT_MOVE_ARG, _)
        {}

        ctor_fun& operator=(BOOST_COPY_ASSIGN_REF(ctor_fun) rhs)
        {
            if (this != &rhs) {
                BOOST_PP_REPEAT(N, HPX_REMOTE_OBJECT_COPY, _)
            }
            return *this;
        }

        ctor_fun& operator=(BOOST_RV_REF(ctor_fun) rhs)
        {
            if (this != &rhs) {
                BOOST_PP_REPEAT(N, HPX_REMOTE_OBJECT_MOVE, _)
            }
            return *this;
        }

        void operator()(void ** p) const
        {
            T * t = new T(BOOST_PP_ENUM_PARAMS(N, a));
            *p = t;
        }

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            BOOST_PP_REPEAT(N, HPX_REMOTE_OBJECT_M1, _)
        }

        BOOST_PP_REPEAT(N, HPX_REMOTE_OBJECT_M2, _)

    private:
        BOOST_COPYABLE_AND_MOVABLE(ctor_fun);
    };

#undef N
#endif

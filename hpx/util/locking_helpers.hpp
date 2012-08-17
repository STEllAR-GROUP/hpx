//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_COMPONENTS_AMR_LOCKING_HELPERS_JAN_24_0225PM)
#define HPX_COMPONENTS_AMR_LOCKING_HELPERS_JAN_24_0225PM

#include <hpx/config.hpp>

#if HPX_LOCK_LIMIT >= 6

#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/comparison/not_equal.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/preprocessor/facilities/empty.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/arithmetic/dec.hpp>

#include <boost/thread/locks.hpp>

// boost doesn't have an overload for more than 5 mutexes
#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/util/preprocessed/locking_helpers.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/locking_helpers_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
        (3, (6, HPX_LOCK_LIMIT,                                               \
        "hpx/util/locking_helpers.hpp"))
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif // HPX_LOCK_LIMIT >= 6

#endif

#else  // BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

#define HPX_ENUM_MUTEX_ARG(z, n, data)                                        \
    BOOST_PP_COMMA_IF(BOOST_PP_DEC(n)) BOOST_PP_CAT(m, n)                     \
    /**/
#define HPX_ENUM_FROM_1_TO_N()                                                \
    BOOST_PP_REPEAT_FROM_TO(1, N, HPX_ENUM_MUTEX_ARG, _)                      \
    /**/

namespace boost { namespace detail
{
    template <BOOST_PP_ENUM_PARAMS(N, typename MutexType)>
    inline unsigned lock_helper(BOOST_PP_ENUM_BINARY_PARAMS(N, MutexType, & m))
    {
        boost::unique_lock<MutexType0> l0(m0);
        if (unsigned const failed_lock = try_lock_internal(HPX_ENUM_FROM_1_TO_N()))
        {
            return failed_lock;
        }
        l0.release();
        return 0;
    }

    template <BOOST_PP_ENUM_PARAMS(N, typename MutexType)>
    inline unsigned try_lock_internal(BOOST_PP_ENUM_BINARY_PARAMS(N, MutexType, & m))
    {
        boost::unique_lock<MutexType0> l0(m0, boost::try_to_lock);
        if (!l0)
        {
            return 1;
        }
        if (unsigned const failed_lock = try_lock_internal(HPX_ENUM_FROM_1_TO_N()))
        {
            return failed_lock + 1;
        }
        l0.release();
        return 0;
    }
}}

#undef HPX_ENUM_MUTEX_ARG
#undef HPX_ENUM_FROM_1_TO_N

#define HPX_ENUM_MUTEX_ARG1(z, n, data)                                       \
    BOOST_PP_CAT(m, n) BOOST_PP_COMMA_IF(BOOST_PP_NOT_EQUAL(n, data))         \
    /**/
#define HPX_ENUM_MUTEX_ARG2(z, n, data)                                       \
    BOOST_PP_COMMA_IF(BOOST_PP_NOT_EQUAL(n, data)) BOOST_PP_CAT(m, n)         \
    /**/

#define HPX_ROTATE_ARGS(m, n, z)                                              \
    BOOST_PP_REPEAT_FROM_TO_ ## z(m, n, HPX_ENUM_MUTEX_ARG1, BOOST_PP_DEC(n)) \
    BOOST_PP_REPEAT_FROM_TO_ ## z(0, m, HPX_ENUM_MUTEX_ARG2, m)               \
    /**/

#define HPX_SET_LOCK_FIRST(n, data) lock_first = (lock_first+n) % data;
#define HPX_LOCK_EMPTY(n, data)

#define HPX_LOCK_CASE(z, n, data)                                             \
    case n:                                                                   \
        lock_first = detail::lock_helper(                                     \
            HPX_ROTATE_ARGS(n, N, z));                                        \
        if (!lock_first)                                                      \
            return;                                                           \
        BOOST_PP_IIF(BOOST_PP_NOT_EQUAL(n, 0),                                \
            HPX_SET_LOCK_FIRST, HPX_LOCK_EMPTY)(n, data)                      \
        break;                                                                \

namespace boost
{
    template <BOOST_PP_ENUM_PARAMS(N, typename MutexType)>
    inline void lock(BOOST_PP_ENUM_BINARY_PARAMS(N, MutexType, & m))
    {
        unsigned lock_first = 0;
        for (;;)
        {
            switch (lock_first)
            {
                BOOST_PP_REPEAT(N, HPX_LOCK_CASE, N)
            }
        }
    }
}

#undef HPX_ROTATE_ARGS
#undef HPX_ENUM_MUTEX_ARG2
#undef HPX_ENUM_MUTEX_ARG1

#undef N

#endif  // BOOST_PP_IS_ITERATING


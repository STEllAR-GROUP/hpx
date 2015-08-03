//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Parts of this have been taken from Eric Niebler's Ranges V3 library, see
// its project home at: https://github.com/ericniebler/range-v3.

#if !defined(HPX_TRAITS_CONCEPTS_JUL_19_2015_0547PM)
#define HPX_TRAITS_CONCEPTS_JUL_19_2015_0547PM

#include <boost/preprocessor/cat.hpp>
#include <boost/static_assert.hpp>

#include <type_traits>

#define HPX_CONCEPT_REQUIRES_(...)                                            \
    int BOOST_PP_CAT(_concept_requires_, __LINE__) = 42,                      \
    typename std::enable_if<                                                  \
        (BOOST_PP_CAT(_concept_requires_, __LINE__) == 43) || (__VA_ARGS__),  \
        int                                                                   \
    >::type = 0                                                               \
    /**/

#define HPX_CONCEPT_REQUIRES(...)                                             \
    template<                                                                 \
        int BOOST_PP_CAT(_concept_requires_, __LINE__) = 42,                  \
        typename std::enable_if<                                              \
            (BOOST_PP_CAT(_concept_requires_, __LINE__) == 43) || (__VA_ARGS__), \
            int                                                               \
        >::type = 0>                                                          \
    /**/

#define HPX_CONCEPT_ASSERT(...)                                               \
    BOOST_STATIC_ASSERT_MSG((__VA_ARGS__), "Concept check failed")            \
    /**/

#endif


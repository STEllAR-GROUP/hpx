//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Parts of this have been taken from Eric Niebler's Ranges V3 library, see
// its project home at: https://github.com/ericniebler/range-v3.

#if !defined(HPX_TRAITS_CONCEPTS_JUL_19_2015_0547PM)
#define HPX_TRAITS_CONCEPTS_JUL_19_2015_0547PM

#include <hpx/util/detail/pp/cat.hpp>

#include <type_traits>

#define HPX_CONCEPT_REQUIRES_(...)                                            \
    int HPX_PP_CAT(_concept_requires_, __LINE__) = 42,                        \
    typename std::enable_if<                                                  \
        (HPX_PP_CAT(_concept_requires_, __LINE__) == 43) || (__VA_ARGS__),    \
        int                                                                   \
    >::type HPX_PP_CAT(_concept_check_, __LINE__) = 0                         \
    /**/

#define HPX_CONCEPT_REQUIRES(...)                                             \
    template<                                                                 \
        int HPX_PP_CAT(_concept_requires_, __LINE__) = 42,                    \
        typename std::enable_if<                                              \
            (HPX_PP_CAT(_concept_requires_, __LINE__) == 43) || (__VA_ARGS__),\
            int                                                               \
        >::type HPX_PP_CAT(_concept_check_, __LINE__) = 0>                    \
    /**/

#define HPX_CONCEPT_ASSERT(...)                                               \
    static_assert((__VA_ARGS__), "Concept check failed")                      \
    /**/

#endif


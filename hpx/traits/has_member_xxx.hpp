//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_HAS_MEMBER_XXX_HPP
#define HPX_TRAITS_HAS_MEMBER_XXX_HPP

#include <boost/mpl/bool.hpp>
#include <boost/preprocessor/cat.hpp>

#include <type_traits>

// This macro creates a boolean unary metafunction which result is
// true if and only if its parameter type has member function with
// MEMBER name (no matter static it is or not). The generated trait
// ends up in a namespace where the macro itself has been placed.
#define HPX_HAS_MEMBER_XXX_TRAIT_DEF(MEMBER)                      \
    namespace BOOST_PP_CAT(BOOST_PP_CAT(has_, MEMBER), _detail)   \
    {                                                             \
        struct helper                                             \
        {                                                         \
            void MEMBER (...);                                    \
        };                                                        \
                                                                  \
        template <typename T>                                     \
        struct helper_composed: T, helper {};                     \
                                                                  \
        template <void (helper::*) (...)>                         \
        struct member_function_holder {};                         \
                                                                  \
        template <typename T, typename Ambiguous =                \
            member_function_holder<&helper::MEMBER> >             \
        struct impl : boost::mpl::true_ {};                       \
                                                                  \
        template <typename T>                                     \
        struct impl<T,                                            \
            member_function_holder<&helper_composed<T>::MEMBER> > \
          : boost::mpl::false_ {};                                \
    }                                                             \
                                                                  \
    template <typename T, typename Enable = void>                 \
    struct BOOST_PP_CAT(has_, MEMBER) : boost::mpl::false_ {};    \
                                                                  \
    template <typename T>                                         \
    struct BOOST_PP_CAT(has_, MEMBER)<T,                          \
        typename std::enable_if<std::is_class<T>::value >::type>  \
          : BOOST_PP_CAT(BOOST_PP_CAT(has_, MEMBER), _detail)     \
            ::impl<T> {};                                         \
/**/

#endif

//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_HAS_MEMBER_XXX_HPP
#define HPX_TRAITS_HAS_MEMBER_XXX_HPP

#include <boost/mpl/bool.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/type_traits/is_class.hpp>
#include <boost/utility/enable_if.hpp>

#define HPX_HAS_MEMBER_XXX_TRAIT_DEF(MEMBER)                      \
    namespace BOOST_PP_CAT(BOOST_PP_CAT(has_, MEMBER), _detail)   \
    {                                                             \
        struct helper                                             \
        {                                                         \
            void MEMBER (...);                                    \
        };                                                        \
                                                                  \
        template <class T>                                        \
        struct helper_composed: T, helper {};                     \
                                                                  \
        template <void (helper::*) (...)>                         \
        struct member_function_holder {};                         \
                                                                  \
        template <class T, class Ambiguous =                      \
            member_function_holder<&helper::MEMBER> >             \
        struct impl: boost::mpl::true_ {};                        \
                                                                  \
        template <class T>                                        \
        struct impl<T,                                            \
            member_function_holder<&helper_composed<T>::MEMBER> > \
          : boost::mpl::false_ {};                                \
    }                                                             \
                                                                  \
    template <class T, class Enable = void>                       \
    struct BOOST_PP_CAT(has_, MEMBER): boost::mpl::false_ {};     \
                                                                  \
    template <class T>                                            \
    struct BOOST_PP_CAT(has_, MEMBER)<T,                          \
        typename boost::enable_if<boost::is_class<T> >::type>:    \
            BOOST_PP_CAT(BOOST_PP_CAT(has_, MEMBER), _detail)     \
            ::impl<T> {};                                         \
/**/


#endif

//  Copyright (c) 2016 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_HAS_XXX_HPP
#define HPX_TRAITS_HAS_XXX_HPP

#include <boost/mpl/bool.hpp>
#include <boost/preprocessor/cat.hpp>

#include <hpx/util/always_void.hpp>

// This macro creates a boolean unary metafunction such that for
// any type X, has_name<X>::value == true if and only if X is a
// class type and has a nested type member x::name. The generated
// trait ends up in a namespace where the macro itself has been
// placed.
#define HPX_HAS_XXX_TRAIT_DEF(Name)                               \
    template <typename T, typename Enable = void>                 \
    struct BOOST_PP_CAT(has_, Name) : boost::mpl::false_ {};      \
                                                                  \
    template <typename T>                                         \
    struct BOOST_PP_CAT(has_, Name)<T,                            \
        typename hpx::util::always_void<typename T::Name>::type>  \
      : boost::mpl::true_ {}                                      \
/**/

#endif

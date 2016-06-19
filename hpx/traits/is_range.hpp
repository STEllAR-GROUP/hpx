//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2016 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_IS_RANGE_HPP
#define HPX_TRAITS_IS_RANGE_HPP

#include <hpx/util/range.hpp>

#include <iterator>
#include <type_traits>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct is_range
      : std::false_type
    {};

    template <typename T>
    struct is_range<
        T,
        typename std::enable_if<
            std::is_same<
                typename util::detail::iterator<T>::type,
                typename util::detail::sentinel<T>::type
            >::value
        >::type
    > : std::true_type
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename R, bool IsRange = is_range<R>::value>
    struct range_traits
    {};

    template <typename R>
    struct range_traits<R, true>
      : std::iterator_traits<
            typename util::detail::iterator<R>::type
        >
    {};
}}

#endif /*HPX_TRAITS_IS_RANGE_HPP*/

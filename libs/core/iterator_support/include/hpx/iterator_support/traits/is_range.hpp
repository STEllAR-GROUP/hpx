//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2016 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_sentinel_for.hpp>

#include <iterator>
#include <type_traits>

namespace hpx { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct is_range : std::false_type
    {
    };

    template <typename T>
    struct is_range<T,
        typename std::enable_if<hpx::traits::is_sentinel_for<
            typename util::detail::sentinel<T>::type,
            typename util::detail::iterator<T>::type>::value>::type>
      : std::true_type
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct range_iterator : util::detail::iterator<T>
    {
    };

    template <typename T, typename Enable = void>
    struct range_sentinel : util::detail::sentinel<T>
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename R, bool IsRange = is_range<R>::value>
    struct range_traits
    {
    };

    template <typename R>
    struct range_traits<R, true>
      : std::iterator_traits<typename util::detail::iterator<R>::type>
    {
        typedef typename util::detail::iterator<R>::type iterator_type;
        typedef typename util::detail::sentinel<R>::type sentinel_type;
    };
}}    // namespace hpx::traits

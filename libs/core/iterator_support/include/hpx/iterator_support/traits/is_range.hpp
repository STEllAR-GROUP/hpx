//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2016 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>

#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx::traits {

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    using is_range = util::detail::is_range<T>;

    template <typename T>
    inline constexpr bool is_range_v = is_range<T>::value;

    ///////////////////////////////////////////////////////////////////////////
    // return whether a given type is a range generator (i.e. exposes supports
    // an iterate function that returns a range)
    template <typename T>
    using is_range_generator = util::detail::is_range_generator<T>;

    template <typename T>
    inline constexpr bool is_range_generator_v = is_range_generator<T>::value;

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct range_iterator : util::detail::iterator<T>
    {
    };

    template <typename Range>
    struct range_iterator<Range, std::enable_if_t<is_range_generator_v<Range>>>
    {
        // clang-format off
        using type = typename range_iterator<decltype(
            hpx::util::iterate(std::declval<Range&>()))>::type;
        // clang-format on
    };

    template <typename T, typename Enable = void>
    struct range_sentinel : util::detail::sentinel<T>
    {
    };

    template <typename T>
    using range_iterator_t = typename range_iterator<T>::type;

    template <typename T>
    using range_sentinel_t = typename range_sentinel<T>::type;

    // return the iterator category encapsulated by the range
    template <typename T>
    using range_category_t = iter_category_t<range_iterator_t<T>>;

    ///////////////////////////////////////////////////////////////////////////
    template <typename R, bool IsRange = is_range<R>::value>
    struct range_traits
    {
    };

    template <typename R>
    struct range_traits<R, true>
      : std::iterator_traits<typename util::detail::iterator<R>::type>
    {
        using iterator_type = typename util::detail::iterator<R>::type;
        using sentinel_type = typename util::detail::sentinel<R>::type;
    };
}    // namespace hpx::traits

//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2016 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_sentinel_for.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx::traits {

    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_CORE_EXPORT template <typename T>
    using is_range = util::detail::is_range<T>;

    HPX_CXX_CORE_EXPORT template <typename T>
    inline constexpr bool is_range_v = is_range<T>::value;

    ///////////////////////////////////////////////////////////////////////////
    // return whether a given type is a range generator (i.e. exposes supports
    // an iterate function that returns a range)
    HPX_CXX_CORE_EXPORT template <typename T>
    using is_range_generator = util::detail::is_range_generator<T>;

    HPX_CXX_CORE_EXPORT template <typename T>
    inline constexpr bool is_range_generator_v = is_range_generator<T>::value;

    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_CORE_EXPORT template <typename T, typename Enable = void>
    struct range_iterator : util::detail::iterator<T>
    {
    };

    HPX_CXX_CORE_EXPORT template <typename Range>
    struct range_iterator<Range, std::enable_if_t<is_range_generator_v<Range>>>
    {
        // clang-format off
        using type = typename range_iterator<decltype(
            hpx::util::iterate(std::declval<Range&>()))>::type;
        // clang-format on
    };

    HPX_CXX_CORE_EXPORT template <typename T, typename Enable = void>
    struct range_sentinel : util::detail::sentinel<T>
    {
    };

    HPX_CXX_CORE_EXPORT template <typename T>
    using range_iterator_t = typename range_iterator<T>::type;

    HPX_CXX_CORE_EXPORT template <typename T>
    using range_sentinel_t = typename range_sentinel<T>::type;

    // return the iterator category encapsulated by the range
    HPX_CXX_CORE_EXPORT template <typename T>
    using range_category_t = iter_category_t<range_iterator_t<T>>;

    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_CORE_EXPORT template <typename R, bool IsRange = is_range<R>::value>
    struct range_traits
    {
    };

    HPX_CXX_CORE_EXPORT template <typename R>
    struct range_traits<R, true>
      : std::iterator_traits<typename util::detail::iterator<R>::type>
    {
        using iterator_type = typename util::detail::iterator<R>::type;
        using sentinel_type = typename util::detail::sentinel<R>::type;
    };

    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_EXPORT template <typename R, typename Enable = void>
    struct is_input_range : std::false_type
    {
    };

    HPX_CXX_EXPORT template <typename R>
    struct is_input_range<R, std::enable_if_t<is_range_v<R>>>
      : std::integral_constant<bool,
            hpx::traits::is_input_iterator_v<hpx::traits::range_iterator_t<R>>>
    {
    };

    HPX_CXX_EXPORT template <typename R>
    using is_input_range_t = typename is_input_range<R>::type;

    HPX_CXX_EXPORT template <typename R>
    inline constexpr bool is_input_range_v = is_input_range<R>::value;

    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_EXPORT template <typename R, typename Enable = void>
    struct is_forward_range : std::false_type
    {
    };

    HPX_CXX_EXPORT template <typename R>
    struct is_forward_range<R, std::enable_if_t<is_range_v<R>>>
      : std::integral_constant<bool,
            hpx::traits::is_forward_iterator_v<
                hpx::traits::range_iterator_t<R>>>
    {
    };

    HPX_CXX_EXPORT template <typename R>
    using is_forward_range_t = typename is_forward_range<R>::type;

    HPX_CXX_EXPORT template <typename R>
    inline constexpr bool is_forward_range_v = is_forward_range<R>::value;

    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_EXPORT template <typename R, typename Enable = void>
    struct is_bidirectional_range : std::false_type
    {
    };

    HPX_CXX_EXPORT template <typename R>
    struct is_bidirectional_range<R, std::enable_if_t<is_range_v<R>>>
      : std::integral_constant<bool,
            hpx::traits::is_bidirectional_iterator_v<
                hpx::traits::range_iterator_t<R>>>
    {
    };

    HPX_CXX_EXPORT template <typename R>
    using is_bidirectional_range_t = typename is_bidirectional_range<R>::type;

    HPX_CXX_EXPORT template <typename R>
    inline constexpr bool is_bidirectional_range_v =
        is_bidirectional_range<R>::value;

    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_EXPORT template <typename R, typename Enable = void>
    struct is_random_access_range : std::false_type
    {
    };

    HPX_CXX_EXPORT template <typename R>
    struct is_random_access_range<R, std::enable_if_t<is_range_v<R>>>
      : std::integral_constant<bool,
            hpx::traits::is_random_access_iterator_v<
                hpx::traits::range_iterator_t<R>>>
    {
    };

    HPX_CXX_EXPORT template <typename R>
    using is_random_access_range_t = typename is_random_access_range<R>::type;

    HPX_CXX_EXPORT template <typename R>
    inline constexpr bool is_random_access_range_v =
        is_random_access_range<R>::value;

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_CXX20_STD_DISABLE_SIZED_RANGE)
    HPX_CXX_EXPORT template <typename R>
    inline constexpr bool disable_sized_range =
        std::ranges::disable_sized_range<R>;
#else
    HPX_CXX_EXPORT template <typename R>
    inline constexpr bool disable_sized_range = false;
#endif

    HPX_CXX_EXPORT template <typename R, typename Enable = void>
    struct is_sized_range : std::false_type
    {
    };

    HPX_CXX_EXPORT template <typename R>
    struct is_sized_range<R,
        std::enable_if_t<is_range_v<R> &&
            !disable_sized_range<std::remove_cv_t<R>> &&
            (util::detail::has_size_member_v<R> ||
                util::detail::has_size_v<R> ||
                std::is_array_v<std::remove_reference_t<R>> ||
                is_sized_sentinel_for_v<range_sentinel_t<R>,
                    range_iterator_t<R>>)>> : std::true_type
    {
    };

    HPX_CXX_EXPORT template <typename R>
    inline constexpr bool is_sized_range_v = is_sized_range<R>::value;
}    // namespace hpx::traits

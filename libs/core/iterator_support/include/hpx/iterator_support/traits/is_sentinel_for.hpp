//  Copyright (c) 2020 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/type_support/always_void.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace traits {

    ///////////////////////////////////////////////////////////////////////////
    // The trait checks whether sentinel Sent is proper for iterator I.
    // There are two requirements for this:
    // 1. iterator I should be an input or output iterator
    // 2. I and S should oblige with the weakly-equality-comparable concept
    template <typename Sent, typename Iter, typename Enable = void>
    struct is_sentinel_for : std::false_type
    {
    };

    template <typename Sent, typename Iter>
    struct is_sentinel_for<Sent, Iter,
        typename std::enable_if_t<is_iterator_v<Iter> &&
            is_weakly_equality_comparable_with<Iter, Sent>::value>>
      : std::true_type
    {
    };

    template <typename Sent, typename Iter>
    inline constexpr bool is_sentinel_for_v =
        is_sentinel_for<Sent, Iter>::value;

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_CXX20_STD_DISABLE_SIZED_SENTINEL_FOR)
    template <typename Sent, typename Iter>
    inline constexpr bool disable_sized_sentinel_for =
        std::disable_sized_sentinel_for<Sent, Iter>;
#else
    template <typename Sent, typename Iter>
    inline constexpr bool disable_sized_sentinel_for = false;
#endif

    template <typename Sent, typename Iter, typename Enable = void>
    struct is_sized_sentinel_for : std::false_type
    {
    };

    template <typename Sent, typename Iter>
    struct is_sized_sentinel_for<Sent, Iter,
        typename util::always_void<
            std::enable_if_t<hpx::traits::is_sentinel_for<Sent, Iter>::value &&
                !disable_sized_sentinel_for<typename std::remove_cv<Sent>::type,
                    typename std::remove_cv<Iter>::type>>,
            typename detail::subtraction_result<Iter, Sent>::type,
            typename detail::subtraction_result<Sent, Iter>::type>::type>
      : std::true_type
    {
    };

    template <typename Sent, typename Iter>
    inline constexpr bool is_sized_sentinel_for_v =
        is_sized_sentinel_for<Sent, Iter>::value;

}}    // namespace hpx::traits

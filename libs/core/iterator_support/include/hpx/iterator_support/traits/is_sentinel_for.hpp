//  Copyright (c) 2020 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/iterator_support/traits/is_iterator.hpp>

#include <type_traits>

namespace hpx::traits {

    ///////////////////////////////////////////////////////////////////////////
    // The trait checks whether sentinel Sent is proper for iterator I.
    // There are two requirements for this:
    // 1. iterator I should be an input or output iterator
    // 2. I and S should oblige with the weakly-equality-comparable concept
    HPX_CXX_CORE_EXPORT template <typename Sent, typename Iter,
        typename Enable = void>
    struct is_sentinel_for : std::false_type
    {
    };

    HPX_CXX_CORE_EXPORT template <typename Sent, typename Iter>
    struct is_sentinel_for<Sent, Iter,
        std::enable_if_t<is_iterator_v<Iter> &&
            is_weakly_equality_comparable_with_v<Iter, Sent>>> : std::true_type
    {
    };

    HPX_CXX_CORE_EXPORT template <typename Sent, typename Iter>
    inline constexpr bool is_sentinel_for_v =
        is_sentinel_for<Sent, Iter>::value;

    ///////////////////////////////////////////////////////////////////////////

    HPX_CXX_CORE_EXPORT template <typename Sent, typename Iter,
        typename Enable = void>
    struct is_sized_sentinel_for : std::false_type
    {
    };

    HPX_CXX_CORE_EXPORT template <typename Sent, typename Iter>
    struct is_sized_sentinel_for<Sent, Iter,
        std::void_t<
            std::enable_if_t<hpx::traits::is_sentinel_for<Sent, Iter>::value &&
                !std::disable_sized_sentinel_for<std::remove_cv_t<Sent>,
                    std::remove_cv_t<Iter>>>,
            detail::subtraction_result_t<Iter, Sent>,
            detail::subtraction_result_t<Sent, Iter>>> : std::true_type
    {
    };

    HPX_CXX_CORE_EXPORT template <typename Sent, typename Iter>
    inline constexpr bool is_sized_sentinel_for_v =
        is_sized_sentinel_for<Sent, Iter>::value;
}    // namespace hpx::traits

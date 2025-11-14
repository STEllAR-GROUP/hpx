//  Copyright (c) 2022-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <iterator>
#include <utility>

namespace hpx::util::detail {

    ///////////////////////////////////////////////////////////////////////
    std::random_access_iterator_tag minimum_category_impl(
        std::random_access_iterator_tag, std::random_access_iterator_tag);
    std::bidirectional_iterator_tag minimum_category_impl(
        std::bidirectional_iterator_tag, std::bidirectional_iterator_tag);
    std::forward_iterator_tag minimum_category_impl(
        std::forward_iterator_tag, std::forward_iterator_tag);
    std::input_iterator_tag minimum_category_impl(
        std::input_iterator_tag, std::input_iterator_tag);

    template <typename T, typename U>
    struct minimum_category
    {
        // clang-format off
        using type = decltype(
            minimum_category_impl(std::declval<T>(), std::declval<U>()));
        // clang-format on
    };
}    // namespace hpx::util::detail

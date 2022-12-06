//  Copyright (c) 022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <iterator>

namespace hpx::util::detail {

    ///////////////////////////////////////////////////////////////////////
    template <typename T, typename U>
    struct minimum_category
    {
        static_assert(sizeof(T) == 0 && sizeof(U) == 0,
            "unknown combination of iterator categories");
    };

    // random_access_iterator_tag
    template <>
    struct minimum_category<std::random_access_iterator_tag,
        std::random_access_iterator_tag>
    {
        using type = std::random_access_iterator_tag;
    };

    template <>
    struct minimum_category<std::random_access_iterator_tag,
        std::bidirectional_iterator_tag>
    {
        using type = std::bidirectional_iterator_tag;
    };

    template <>
    struct minimum_category<std::bidirectional_iterator_tag,
        std::random_access_iterator_tag>
    {
        using type = std::bidirectional_iterator_tag;
    };

    template <>
    struct minimum_category<std::random_access_iterator_tag,
        std::forward_iterator_tag>
    {
        using type = std::forward_iterator_tag;
    };

    template <>
    struct minimum_category<std::forward_iterator_tag,
        std::random_access_iterator_tag>
    {
        using type = std::forward_iterator_tag;
    };

    template <>
    struct minimum_category<std::random_access_iterator_tag,
        std::input_iterator_tag>
    {
        using type = std::input_iterator_tag;
    };

    template <>
    struct minimum_category<std::input_iterator_tag,
        std::random_access_iterator_tag>
    {
        using type = std::input_iterator_tag;
    };

    // bidirectional_iterator_tag
    template <>
    struct minimum_category<std::bidirectional_iterator_tag,
        std::bidirectional_iterator_tag>
    {
        using type = std::bidirectional_iterator_tag;
    };

    template <>
    struct minimum_category<std::bidirectional_iterator_tag,
        std::forward_iterator_tag>
    {
        using type = std::forward_iterator_tag;
    };

    template <>
    struct minimum_category<std::forward_iterator_tag,
        std::bidirectional_iterator_tag>
    {
        using type = std::forward_iterator_tag;
    };

    template <>
    struct minimum_category<std::bidirectional_iterator_tag,
        std::input_iterator_tag>
    {
        using type = std::input_iterator_tag;
    };

    template <>
    struct minimum_category<std::input_iterator_tag,
        std::bidirectional_iterator_tag>
    {
        using type = std::input_iterator_tag;
    };

    // forward_iterator_tag
    template <>
    struct minimum_category<std::forward_iterator_tag,
        std::forward_iterator_tag>
    {
        using type = std::forward_iterator_tag;
    };

    template <>
    struct minimum_category<std::input_iterator_tag, std::forward_iterator_tag>
    {
        using type = std::input_iterator_tag;
    };

    template <>
    struct minimum_category<std::forward_iterator_tag, std::input_iterator_tag>
    {
        using type = std::input_iterator_tag;
    };

    // input_iterator_tag
    template <>
    struct minimum_category<std::input_iterator_tag, std::input_iterator_tag>
    {
        using type = std::input_iterator_tag;
    };
}    // namespace hpx::util::detail

//  Copyright (c) 2020 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/type_support/always_void.hpp>
#include <utility>

namespace hpx { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        template <typename T, typename U, typename Enable = void>
        struct equality_result
        {
        };

        template <typename T, typename U>
        struct equality_result<T, U,
            typename util::always_void<decltype(
                std::declval<const T&>() == std::declval<const U&>())>::type>
        {
            using type =
                decltype(std::declval<const T&>() == std::declval<const U&>());
        };

        template <typename T, typename U, typename Enable = void>
        struct inequality_result
        {
        };

        template <typename T, typename U>
        struct inequality_result<T, U,
            typename util::always_void<decltype(
                std::declval<const T&>() != std::declval<const U&>())>::type>
        {
            using type =
                decltype(std::declval<const T&>() != std::declval<const U&>());
        };
    }    // namespace detail

    template <typename Iter, typename Sent, typename Enable = void>
    struct is_sentinel_for : std::false_type
    {
    };

    template <typename Iter, typename Sent>
    struct is_sentinel_for<Iter, Sent,
        typename util::always_void<typename detail::equality_result<Iter, Sent>::type,
            typename detail::inequality_result<Iter, Sent>::type>::type>
      : std::true_type
    {
    };
}}    // namespace hpx::traits

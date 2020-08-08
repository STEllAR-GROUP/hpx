//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c) 2019 Austin McCartney
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/type_support/always_void.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace traits {
    namespace detail {

        ///////////////////////////////////////////////////////////////////////
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

        ///////////////////////////////////////////////////////////////////////
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

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename U, typename Enable = void>
        struct is_weakly_equality_comparable_with : std::false_type
        {
        };

        template <typename T, typename U>
        struct is_weakly_equality_comparable_with<T, U,
            typename util::always_void<
                typename detail::equality_result<T, U>::type,
                typename detail::equality_result<U, T>::type,
                typename detail::inequality_result<T, U>::type,
                typename detail::inequality_result<U, T>::type>::type>
          : std::true_type
        {
        };

    }    // namespace detail

    template <typename T, typename U>
    struct is_weakly_equality_comparable_with
      : detail::is_weakly_equality_comparable_with<typename std::decay<T>::type,
            typename std::decay<U>::type>
    {
    };

    // for now is_equality_comparable is equivalent to its weak version
    template <typename T, typename U>
    struct is_equality_comparable_with
      : detail::is_weakly_equality_comparable_with<typename std::decay<T>::type,
            typename std::decay<U>::type>
    {
    };
}}    // namespace hpx::traits

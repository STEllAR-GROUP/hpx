//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <array>
#include <iterator>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx::traits {
    ///////////////////////////////////////////////////////////////////////////
    // Iterators are contiguous if they are pointers (without concepts we have
    // no generic way of determining whether an iterator is contiguous)

    namespace detail {

        template <typename Iter>
        using iter_value_type_t =
            typename std::iterator_traits<Iter>::value_type;

        template <typename T>
        inline constexpr bool has_valid_array_v =
            std::is_copy_assignable_v<T> && !std::is_function_v<T>;

        template <typename Iter, typename Enable = void>
        struct is_std_array_iterator : std::false_type
        {
        };

        template <typename Iter>
        struct is_std_array_iterator<Iter,
            std::enable_if_t<has_valid_array_v<iter_value_type_t<Iter>>>>
          : std::bool_constant<(
                std::is_same_v<
                    typename std::array<iter_value_type_t<Iter>, 1>::iterator,
                    Iter> ||
                std::is_same_v<typename std::array<iter_value_type_t<Iter>,
                                   1>::const_iterator,
                    Iter>)>
        {
        };

        template <typename T>
        inline constexpr bool has_valid_vector_v =
            std::is_copy_assignable_v<T> && !std::is_function_v<T>;

        template <typename T, typename Enable = void>
        struct is_std_vector_iterator : std::false_type
        {
        };

        template <typename Iter>
        struct is_std_vector_iterator<Iter,
            std::enable_if_t<has_valid_vector_v<iter_value_type_t<Iter>>>>
          : std::bool_constant<
                std::is_same_v<
                    typename std::vector<iter_value_type_t<Iter>>::iterator,
                    Iter> ||
                std::is_same_v<typename std::vector<
                                   iter_value_type_t<Iter>>::const_iterator,
                    Iter>>
        {
        };

        template <typename T>
        inline constexpr bool has_valid_basic_string_v =
            std::is_copy_assignable_v<T> && !std::is_function_v<T> &&
            std::is_trivial_v<T>;

        template <typename T, typename Enable = void>
        struct is_std_basic_string_iterator : std::false_type
        {
        };

        template <typename Iter>
        struct is_std_basic_string_iterator<Iter,
            std::enable_if_t<has_valid_basic_string_v<iter_value_type_t<Iter>>>>
          : std::bool_constant<
                std::is_same_v<typename std::basic_string<
                                   iter_value_type_t<Iter>>::iterator,
                    Iter> ||
                std::is_same_v<typename std::basic_string<
                                   iter_value_type_t<Iter>>::const_iterator,
                    Iter>>
        {
        };

        template <typename Iter>
        struct is_known_contiguous_iterator
          : std::bool_constant<is_std_array_iterator<Iter>::value ||
                is_std_vector_iterator<Iter>::value ||
                is_std_basic_string_iterator<Iter>::value>
        {
        };
    }    // namespace detail

    template <typename Iter>
    struct is_contiguous_iterator
      : std::bool_constant<std::is_pointer_v<Iter> ||
            detail::is_known_contiguous_iterator<Iter>::value>
    {
    };

    template <typename Iter>
    using is_contiguous_iterator_t =
        typename is_contiguous_iterator<Iter>::type;

    template <typename Iter>
    inline constexpr bool is_contiguous_iterator_v =
        is_contiguous_iterator<Iter>::value;
}    // namespace hpx::traits

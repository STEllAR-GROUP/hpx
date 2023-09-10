//  Copyright (c) 2007-2022 Hartmut Kaiser
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

        template <typename T, typename Enable = void>
        struct is_known_contiguous_iterator : std::false_type
        {
        };

        template <typename Iter>
        struct is_known_contiguous_iterator<Iter,
            std::enable_if_t<!std::is_array_v<Iter>>>
          : std::bool_constant<
                std::is_same_v<    // for std::vector
                    typename std::vector<iter_value_type_t<Iter>>::iterator,
                    Iter> ||
                std::is_same_v<typename std::vector<
                                   iter_value_type_t<Iter>>::const_iterator,
                    Iter> ||    // for std::array
                std::is_same_v<
                    typename std::array<iter_value_type_t<Iter>, 1>::iterator,
                    Iter> ||
                std::is_same_v<typename std::array<iter_value_type_t<Iter>,
                                   1>::const_iterator,
                    Iter> ||    // for std::string
                std::is_same_v<typename std::string::iterator, Iter> ||
                std::is_same_v<typename std::string::const_iterator, Iter>>
        {
        };
    }    // namespace detail

    template <typename Iter,
        bool not_known_contiguous_iterator =
    // When _GLIBCXX_DEBUG is defined vectors are contiguous, but the iterators
    // are not plain pointers.
#if defined(_GLIBCXX_DEBUG)
            false
#else
            detail::is_known_contiguous_iterator<Iter>::value
#endif
        >
    struct is_contiguous_iterator : std::is_pointer<Iter>::type
    {
    };

    template <typename Iter>
    struct is_contiguous_iterator<Iter, true> : std::true_type
    {
    };

    template <typename Iter>
    using is_contiguous_iterator_t =
        typename is_contiguous_iterator<Iter>::type;

    template <typename Iter>
    inline constexpr bool is_contiguous_iterator_v =
        is_contiguous_iterator<Iter>::value;
}    // namespace hpx::traits

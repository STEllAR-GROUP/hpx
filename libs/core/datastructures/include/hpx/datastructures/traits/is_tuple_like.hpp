//  Copyright (c) 2017 Denis Blank
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <type_traits>

#include <hpx/datastructures/tuple.hpp>
#include <hpx/type_support/always_void.hpp>

namespace hpx { namespace traits {
    namespace detail {
        template <typename T, typename Enable = void>
        struct is_tuple_like_impl : std::false_type
        {
        };

        template <typename T>
        struct is_tuple_like_impl<T,
            typename util::always_void<decltype(
                hpx::tuple_size<T>::value)>::type> : std::true_type
        {
        };
    }    // namespace detail

    /// Deduces to a true type if the given parameter T
    /// has a specific tuple like size.
    template <typename T>
    struct is_tuple_like
      : detail::is_tuple_like_impl<typename std::remove_cv<T>::type>
    {
    };
}}    // namespace hpx::traits

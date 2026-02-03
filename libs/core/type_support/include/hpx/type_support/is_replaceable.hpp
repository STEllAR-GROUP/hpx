//  Copyright (c) 2025 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <type_traits>

#include <hpx/type_support/is_trivially_relocatable.hpp>

namespace hpx::experimental {

    HPX_CXX_EXPORT template <typename T>
    struct is_replaceable
      : std::bool_constant<std::is_move_constructible_v<T> &&
            std::is_move_assignable_v<T> && is_trivially_relocatable_v<T>>
    {
    };

    HPX_CXX_EXPORT template <typename T>
    inline constexpr bool is_replaceable_v = is_replaceable<T>::value;

}    // namespace hpx::experimental

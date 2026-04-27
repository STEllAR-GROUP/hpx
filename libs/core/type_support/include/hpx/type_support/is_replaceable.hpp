//  Copyright (c) 2025 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx::experimental {

// std::is_replaceable is a library trait introduced by P2786R13.
// Guard its use on the library feature-test macro __cpp_lib_trivially_relocatable
// rather than the core-language macro __cpp_trivial_relocatability, since the
// standard library may not ship the trait even when the language keyword is
// available (or vice versa).
#if defined(__cpp_lib_trivially_relocatable)
    HPX_CXX_CORE_EXPORT template <typename T>
    struct is_replaceable : std::is_replaceable<T>
    {
    };
#else
    HPX_CXX_CORE_EXPORT template <typename T>
    struct is_replaceable
      : std::bool_constant<std::is_object_v<T> && !std::is_const_v<T> &&
            !std::is_volatile_v<T> &&
            (std::is_scalar_v<T> ||
                ((std::is_class_v<T> || std::is_union_v<T>) &&
                    std::is_trivially_move_constructible_v<T> &&
                    std::is_trivially_move_assignable_v<T> &&
                    std::is_trivially_destructible_v<T>) )>
    {
    };

    HPX_CXX_CORE_EXPORT template <typename T>
    struct is_replaceable<T[]> : std::false_type
    {
    };

    HPX_CXX_CORE_EXPORT template <typename T, std::size_t N>
    struct is_replaceable<T[N]> : std::false_type
    {
    };
#endif

    HPX_CXX_CORE_EXPORT template <typename T>
    inline constexpr bool is_replaceable_v = is_replaceable<T>::value;

}    // namespace hpx::experimental

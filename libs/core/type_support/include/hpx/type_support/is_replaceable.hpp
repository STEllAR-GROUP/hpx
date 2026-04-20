//  Copyright (c) 2025 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstddef>
#include <type_traits>

#include <hpx/type_support/is_trivially_relocatable.hpp>

namespace hpx::experimental {

// P2786R13 defines a single feature-test macro __cpp_trivial_relocatability
// that covers both the core-language keyword and the associated library traits
// (std::is_trivially_relocatable, std::is_replaceable, std::relocate_at).
// The language and library features are bundled in one proposal, so guarding on
// this macro is sufficient to confirm that std::is_replaceable is available.
#if defined(__cpp_trivial_relocatability)
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

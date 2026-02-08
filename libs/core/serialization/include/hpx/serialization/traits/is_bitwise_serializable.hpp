//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2022-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/serialization/config/defines.hpp>
#include <hpx/serialization/macros.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#include <type_traits>

namespace hpx::traits {

#if !defined(HPX_SERIALIZATION_HAVE_ALLOW_RAW_POINTER_SERIALIZATION)
    HPX_CXX_CORE_EXPORT template <typename T, typename Enable = void>
    struct is_bitwise_serializable
      : std::integral_constant<bool,
            (std::is_trivially_copy_assignable_v<T> ||
                (std::is_copy_assignable_v<T> &&
                    std::is_trivially_copy_constructible_v<T>) ) &&
                !std::is_pointer_v<T>>
    {
    };
#else
    HPX_CXX_CORE_EXPORT template <typename T, typename Enable = void>
    struct is_bitwise_serializable
      : std::integral_constant<bool,
            std::is_trivially_copy_assignable_v<T> ||
                (std::is_copy_assignable_v<T> &&
                    std::is_trivially_copy_constructible_v<T>)>
    {
    };
#endif

    HPX_CXX_CORE_EXPORT template <typename T>
    inline constexpr bool is_bitwise_serializable_v =
        is_bitwise_serializable<T>::value;

}    // namespace hpx::traits

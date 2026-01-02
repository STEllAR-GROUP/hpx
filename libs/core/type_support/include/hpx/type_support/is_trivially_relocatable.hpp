//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <type_traits>

namespace hpx::experimental {

    // All trivially copyable types are trivially relocatable
    // Other types should default to false.
    HPX_CXX_CORE_EXPORT template <typename T>
    struct is_trivially_relocatable : std::is_trivially_copyable<T>
    {
    };

    // References are not trivially relocatable
    HPX_CXX_CORE_EXPORT template <typename T>
    struct is_trivially_relocatable<T&> : std::false_type
    {
    };

    // Temporary objects are not trivially relocatable
    HPX_CXX_CORE_EXPORT template <typename T>
    struct is_trivially_relocatable<T&&> : std::false_type
    {
    };

    // Constness, Volatility, Arrays are ignored
    HPX_CXX_CORE_EXPORT template <typename T>
    struct is_trivially_relocatable<T const> : is_trivially_relocatable<T>
    {
    };

    HPX_CXX_CORE_EXPORT template <typename T>
    struct is_trivially_relocatable<T volatile> : is_trivially_relocatable<T>
    {
    };

    HPX_CXX_CORE_EXPORT template <typename T>
    struct is_trivially_relocatable<T const volatile>
      : is_trivially_relocatable<T>
    {
    };

    HPX_CXX_CORE_EXPORT template <typename T>
    struct is_trivially_relocatable<T[]> : is_trivially_relocatable<T>
    {
    };

    HPX_CXX_CORE_EXPORT template <typename T, int N>
    struct is_trivially_relocatable<T[N]> : is_trivially_relocatable<T>
    {
    };

    HPX_CXX_CORE_EXPORT template <typename T, int N>
    struct is_trivially_relocatable<T const[N]> : is_trivially_relocatable<T>
    {
    };

    HPX_CXX_CORE_EXPORT template <typename T, int N>
    struct is_trivially_relocatable<T volatile[N]> : is_trivially_relocatable<T>
    {
    };

    HPX_CXX_CORE_EXPORT template <typename T, int N>
    struct is_trivially_relocatable<T const volatile[N]>
      : is_trivially_relocatable<T>
    {
    };

    HPX_CXX_CORE_EXPORT template <typename T>
    inline constexpr bool is_trivially_relocatable_v =
        is_trivially_relocatable<T>::value;
}    // namespace hpx::experimental

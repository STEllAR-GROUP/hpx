//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/serialization/config/defines.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#include <type_traits>

namespace hpx::traits {

#if !defined(HPX_SERIALIZATION_HAVE_ALLOW_RAW_POINTER_SERIALIZATION)
    template <typename T>
    struct is_bitwise_serializable : std::is_arithmetic<T>
    {
    };
#else
    template <typename T>
    struct is_bitwise_serializable
      : std::integral_constant<bool,
            std::is_arithmetic_v<T> || std::is_pointer_v<T>>
    {
    };
#endif

    template <typename T>
    inline constexpr bool is_bitwise_serializable_v =
        is_bitwise_serializable<T>::value;

}    // namespace hpx::traits

#define HPX_IS_BITWISE_SERIALIZABLE(T)                                         \
    namespace hpx::traits {                                                    \
        template <>                                                            \
        struct is_bitwise_serializable<T> : std::true_type                     \
        {                                                                      \
        };                                                                     \
    }                                                                          \
    /**/

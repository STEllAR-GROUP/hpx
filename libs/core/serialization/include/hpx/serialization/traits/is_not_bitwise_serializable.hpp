//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <type_traits>

namespace hpx { namespace traits {

    // This traits can be used in systems that assume that all types are bitwise
    // serializable by default (like SHAD), while still enforcing normal
    // serialization for a given type.

#if defined(HPX_SERIALIZATION_HAVE_ALL_TYPES_ARE_BITWISE_SERIALIZABLE)
    template <typename T>
    struct is_not_bitwise_serializable : std::is_abstract<T>
    {
    };
#else
    template <typename T>
    struct is_not_bitwise_serializable : std::true_type
    {
    };
#endif

    template <typename T>
    HPX_INLINE_CONSTEXPR_VARIABLE bool is_not_bitwise_serializable_v =
        is_not_bitwise_serializable<T>::value;

}}    // namespace hpx::traits

#define HPX_IS_NOT_BITWISE_SERIALIZABLE(T)                                     \
    namespace hpx { namespace traits {                                         \
            template <>                                                        \
            struct is_not_bitwise_serializable<T> : std::true_type             \
            {                                                                  \
            };                                                                 \
        }                                                                      \
    }                                                                          \
    /**/

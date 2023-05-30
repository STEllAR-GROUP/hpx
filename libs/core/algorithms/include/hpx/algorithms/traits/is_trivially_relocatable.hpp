//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <type_traits>

// Macro to specialize template for given type
#define HPX_DECLARE_TRIVIALLY_RELOCATABLE(T)                                   \
    namespace hpx {                                                            \
        template <>                                                            \
        struct is_trivially_relocatable<T> : std::true_type                    \
        {                                                                      \
        };                                                                     \
    }

namespace hpx {

    template <typename T, typename = void>
    struct is_trivially_relocatable : std::false_type
    {
    };

    // All trivially copyable types are trivially relocatable
    template <typename T>
    struct is_trivially_relocatable<T,
        std::enable_if_t<std::is_trivially_copyable_v<T>>> : std::true_type
    {
    };

    template <typename T>
    inline constexpr bool is_trivially_relocatable_v =
        is_trivially_relocatable<T>::value;
}    // namespace hpx

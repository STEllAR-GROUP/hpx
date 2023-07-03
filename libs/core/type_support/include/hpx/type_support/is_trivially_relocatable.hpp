//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <type_traits>

// Macro to specialize template for given type
#define HPX_DECLARE_TRIVIALLY_RELOCATABLE(T)                                   \
    namespace hpx {                                                            \
        template <>                                                            \
        struct is_trivially_relocatable<T> : std::true_type                    \
        {                                                                      \
        };                                                                     \
    }

#define HPX_DECLARE_TRIVIALLY_RELOCATABLE_TEMPLATE(T)                          \
    namespace hpx {                                                            \
        template <typename... K>                                               \
        struct is_trivially_relocatable<T<K...>> : std::true_type              \
        {                                                                      \
        };                                                                     \
    }

#define HPX_DECLARE_TRIVIALLY_RELOCATABLE_TEMPLATE_IF(T, Condition)            \
    namespace hpx {                                                            \
        template <typename... K>                                               \
        struct is_trivially_relocatable<T<K...>> : Condition<K...>             \
        {                                                                      \
        };                                                                     \
    }

namespace hpx {

    template <typename T>
    // All trivially copyable types are trivially relocatable
    // Other types should default to false.
    struct is_trivially_relocatable : std::is_trivially_copyable<T>
    {
    };

    template <typename T>
    inline constexpr bool is_trivially_relocatable_v =
        is_trivially_relocatable<T>::value;
}    // namespace hpx

//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <type_traits>

namespace hpx {

    template <typename T>
    struct is_relocatable
    {
        static constexpr bool value =
            std::is_move_constructible_v<T> && std::is_destructible_v<T>;
    };

    template <typename T>
    inline constexpr bool is_relocatable_v = is_relocatable<T>::value;
}    // namespace hpx

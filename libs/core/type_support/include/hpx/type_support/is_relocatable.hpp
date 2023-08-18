//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <type_traits>

namespace hpx {

    template <typename T>
    struct is_relocatable : std::is_move_constructible<T>
    {
    };

    // ToTp(FromTp&&) must be well-formed
    template <typename ToTp, typename FromTp>
    struct is_relocatable_from
      : std::bool_constant<
            std::is_constructible_v<std::remove_cv_t<ToTp>, FromTp> &&
            std::is_same_v<std::decay_t<ToTp>, std::decay_t<FromTp>>>
    {
    };

    template <typename T>
    inline constexpr bool is_relocatable_v = is_relocatable<T>::value;

    template <typename ToTp, typename FromTp>
    inline constexpr bool is_relocatable_from_v =
        is_relocatable_from<ToTp, FromTp>::value;
}    // namespace hpx

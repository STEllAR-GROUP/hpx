//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <type_traits>

namespace hpx::experimental {

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename T>
    struct is_relocatable
      : std::bool_constant<std::is_move_constructible_v<T> &&
            std::is_object_v<T>>
    {
    };

    // ToTp(FromTp&&) must be well-formed
    HPX_CORE_MODULE_EXPORT_EXTERN template <typename ToTp, typename FromTp>
    struct is_relocatable_from
      : std::bool_constant<
            std::is_constructible_v<std::remove_cv_t<ToTp>, FromTp> &&
            std::is_same_v<std::remove_cv_t<ToTp>, std::remove_cv_t<FromTp>>>
    {
    };

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename T>
    inline constexpr bool is_relocatable_v = is_relocatable<T>::value;

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename ToTp, typename FromTp>
    inline constexpr bool is_relocatable_from_v =
        is_relocatable_from<ToTp, FromTp>::value;
}    // namespace hpx::experimental

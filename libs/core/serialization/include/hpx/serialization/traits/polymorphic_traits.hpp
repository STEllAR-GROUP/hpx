//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2022-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/modules/concepts.hpp>
#include <hpx/modules/preprocessor.hpp>
#include <hpx/serialization/macros.hpp>

#include <type_traits>

namespace hpx::traits {

    namespace detail {

        HPX_HAS_XXX_TRAIT_DEF(HPX_CXX_CORE_EXPORT, serialized_with_id)
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(
            HPX_CXX_CORE_EXPORT, hpx_serialization_get_name)
    }    // namespace detail

    HPX_CXX_CORE_EXPORT template <typename T>
    struct is_intrusive_polymorphic : detail::has_hpx_serialization_get_name<T>
    {
    };

    HPX_CXX_CORE_EXPORT template <typename T>
    inline constexpr bool is_intrusive_polymorphic_v =
        is_intrusive_polymorphic<T>::value;

    HPX_CXX_CORE_EXPORT template <typename T>
    struct is_nonintrusive_polymorphic : std::false_type
    {
    };

    HPX_CXX_CORE_EXPORT template <typename T>
    inline constexpr bool is_nonintrusive_polymorphic_v =
        is_nonintrusive_polymorphic<T>::value;

    HPX_CXX_CORE_EXPORT template <typename T>
    struct is_serialized_with_id : detail::has_serialized_with_id<T>
    {
    };

    HPX_CXX_CORE_EXPORT template <typename T>
    inline constexpr bool is_serialized_with_id_v =
        is_serialized_with_id<T>::value;
}    // namespace hpx::traits

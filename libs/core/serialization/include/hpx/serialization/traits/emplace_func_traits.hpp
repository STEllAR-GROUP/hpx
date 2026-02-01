//  Copyright (c) 2026 Ujjwal Shekhar
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

    HPX_CXX_EXPORT template <typename Collection, class = void>
    struct has_emplace : std::false_type
    {
    };
    
    HPX_CXX_EXPORT template <typename Collection>
    struct has_emplace<Collection,
        std::void_t<decltype(std::declval<Collection>().emplace(
            std::declval<typename Collection::value_type>()))>>
      : std::true_type
    {
    };

    HPX_CXX_EXPORT template <typename Collection>
    inline constexpr bool has_emplace_v = has_emplace<Collection>::value;

    HPX_CXX_EXPORT template <typename Collection, class = void>
    struct has_emplace_back : std::false_type
    {
    };
    
    HPX_CXX_EXPORT template <typename Collection>
    struct has_emplace_back<Collection,
        std::void_t<decltype(std::declval<Collection>().emplace_back(
            std::declval<typename Collection::value_type>()))>>
      : std::true_type
    {
    };

    HPX_CXX_EXPORT template <typename Collection>
    inline constexpr bool has_emplace_back_v = has_emplace_back<Collection>::value;

}    // namespace hpx::traits

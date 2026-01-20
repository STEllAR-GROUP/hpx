//  Copyright (c) 2016-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <type_traits>

namespace hpx::util {

    HPX_CXX_EXPORT template <bool Enable, typename C1, typename C2>
    struct lazy_conditional : std::conditional_t<Enable, C1, C2>
    {
    };

    HPX_CXX_EXPORT template <bool Enable, typename C1, typename C2>
    using lazy_conditional_t = typename lazy_conditional<Enable, C1, C2>::type;

    HPX_CXX_EXPORT template <bool Enable, typename C1, typename C2>
    inline constexpr bool lazy_conditional_v =
        lazy_conditional<Enable, C1, C2>::value;
}    // namespace hpx::util

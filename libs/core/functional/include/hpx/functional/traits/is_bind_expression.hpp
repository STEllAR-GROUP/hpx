//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <functional>

namespace hpx {

    template <typename T>
    struct is_bind_expression : std::is_bind_expression<T>
    {
    };

    template <typename T>
    struct is_bind_expression<T const> : is_bind_expression<T>
    {
    };

    template <typename T>
    inline constexpr bool is_bind_expression_v = is_bind_expression<T>::value;
}    // namespace hpx

namespace hpx::traits {

    template <typename T>
    using is_bind_expression HPX_DEPRECATED_V(1, 8,
        "hpx::traits::is_bind_expression is deprecated, use "
        "hpx::is_bind_expression instead") = hpx::is_bind_expression<T>;

    template <typename T>
    HPX_DEPRECATED_V(1, 8,
        "hpx::traits::is_bind_expression_v is deprecated, use "
        "hpx::is_bind_expression_v instead")
    inline constexpr bool is_bind_expression_v = hpx::is_bind_expression_v<T>;
}    // namespace hpx::traits

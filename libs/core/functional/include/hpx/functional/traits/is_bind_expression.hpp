//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>

#include <functional>

namespace hpx { namespace traits {
    template <typename T>
    struct is_bind_expression : std::is_bind_expression<T>
    {
    };

    template <typename T>
    struct is_bind_expression<T const> : is_bind_expression<T>
    {
    };

    template <typename T>
    HPX_INLINE_CONSTEXPR_VARIABLE bool is_bind_expression_v =
        is_bind_expression<T>::value;
}}    // namespace hpx::traits

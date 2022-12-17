//  Copyright (c) 2021-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

namespace hpx::util {

    struct empty_function
    {
        template <typename... Ts>
        constexpr void operator()(Ts&&...) const noexcept
        {
        }
    };
}    // namespace hpx::util

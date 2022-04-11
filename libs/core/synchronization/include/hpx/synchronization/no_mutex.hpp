//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

namespace hpx {

    struct no_mutex
    {
        constexpr void lock() noexcept {}

        constexpr bool try_lock() noexcept
        {
            return true;
        }

        constexpr void unlock() noexcept {}
    };
}    // namespace hpx

namespace hpx::lcos::local {

    using no_mutex HPX_DEPRECATED_V(1, 8,
        "hpx::lcos::local::no_mutex is deprecated, use hpx::no_mutex instead") =
        hpx::no_mutex;
}    // namespace hpx::lcos::local

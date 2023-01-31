//  Copyright (c) 2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <cstddef>

namespace hpx::util {

    constexpr std::size_t calculate_fanout(
        std::size_t size, std::size_t local_fanout) noexcept
    {
        if (size == 0 || local_fanout == 0)
            return 1;
        if (size <= local_fanout)
            return size;

        std::size_t fanout = 1;
        size -= local_fanout;
        while (fanout < size)
        {
            fanout *= local_fanout;
        }
        return fanout;
    }
}    // namespace hpx::util

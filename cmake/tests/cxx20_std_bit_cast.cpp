//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Test if std::bit_cast is available

#include <bit>
#include <cstdint>

int main()
{
    float a = 42.0;
    std::uint32_t i = std::bit_cast<std::uint32_t>(a);
    (void) i;
    return 0;
}

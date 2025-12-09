////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <ranges>

int main()
{
    constexpr bool b = std::ranges::disable_sized_range<void>;
    (void) b;

    return 0;
}

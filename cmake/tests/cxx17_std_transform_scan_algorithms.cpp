//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <numeric>

int main()
{
    int a[]{1, 10, 100, 1000};
    std::transform_inclusive_scan(
        a, a + 4, a, [](int v1, int v2) { return v1 + v2; },
        [](int v) { return 2 * v; });

    std::transform_exclusive_scan(
        a, a + 4, a, 0, [](int v1, int v2) { return v1 + v2; },
        [](int v) { return 2 * v; });
}

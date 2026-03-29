//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// test for availability of std::execution::<policies> (C++20)

#include <execution>

int main()
{
    std::execution::unsequenced_policy unseq;
    (void) unseq;

    return 0;
}

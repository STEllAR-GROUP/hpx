//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// test for availability of std::execution::<policies> (C++17)

#include <execution>

int main()
{
    std::execution::sequenced_policy seq;
    (void) seq;

    std::execution::parallel_policy par;
    (void) par;

    std::execution::parallel_unsequenced_policy par_unseq;
    (void) par_unseq;

    return 0;
}

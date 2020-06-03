//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/datastructures/optional.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/modules/testing.hpp>

int main()
{
    hpx::util::optional<int> x;
    int y = 42;
    x = y;

    HPX_TEST(x.has_value());
    HPX_TEST_EQ(*x, 42);

    return hpx::util::report_errors();
}

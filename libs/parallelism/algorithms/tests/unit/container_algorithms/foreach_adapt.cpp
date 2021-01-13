//  Copyright (c) 2020 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/modules/testing.hpp>
#include "iter_sent.hpp"

#include <cstdint>
#include <iostream>

void myfunction(int i)
{
    std::cout << ' ' << i;
}

void test_invoke_projected()
{
    iterator<std::int64_t> iter = hpx::ranges::for_each(hpx::execution::seq,
        iterator<std::int64_t>{0}, sentinel<int64_t>{100}, myfunction);

    HPX_TEST_EQ(*iter, std::int64_t(100));

    iter = hpx::ranges::for_each(hpx::execution::par, iterator<std::int64_t>{0},
        sentinel<int64_t>{100}, myfunction);

    HPX_TEST_EQ(*iter, std::int64_t(100));
}

void test_begin_end_iterator()
{
    iterator<std::int64_t> iter = hpx::ranges::for_each(hpx::execution::seq,
        iterator<std::int64_t>{0}, sentinel<int64_t>{100}, &myfunction);

    HPX_TEST_EQ(*iter, std::int64_t(100));

    iter = hpx::ranges::for_each(hpx::execution::par, iterator<std::int64_t>{0},
        sentinel<int64_t>{100}, &myfunction);

    HPX_TEST_EQ(*iter, std::int64_t(100));
}

int main()
{
    test_begin_end_iterator();
    test_invoke_projected();

    return hpx::util::report_errors();
}

//  Copyright (c) 2026 Arpit Khandelwal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/util/stride.hpp>

#include <vector>

void test_stride()
{
    std::vector<int> v = {1, 2, 3, 4, 5, 6};
    // stride(2) -> 1, 3, 5
    auto result = hpx::views::stride(v, 2);

    auto it = result.begin();
    auto end = result.end();

    HPX_TEST_EQ(*it, 1);
    ++it;
    HPX_TEST_EQ(*it, 3);
    ++it;
    HPX_TEST_EQ(*it, 5);
    ++it;
    HPX_TEST(it == end);
}

void test_stride_1()
{
    std::vector<int> v = {1, 2, 3};
    auto result = hpx::views::stride(v, 1);

    auto it = result.begin();
    HPX_TEST_EQ(*it, 1);
    ++it;
    HPX_TEST_EQ(*it, 2);
    ++it;
    HPX_TEST_EQ(*it, 3);
    ++it;
    HPX_TEST(it == result.end());
}

int main(int, char*[])
{
    test_stride();
    test_stride_1();

    return hpx::util::report_errors();
}

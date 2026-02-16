//  Copyright (c) 2026 Arpit Khandelwal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/util/repeat.hpp>

#include <iterator>
#include <vector>

void test_repeat_finite()
{
    auto result = hpx::views::repeat(42, 3);

    auto it = result.begin();
    auto end = result.end();

    HPX_TEST_EQ(*it, 42);
    ++it;
    HPX_TEST_EQ(*it, 42);
    ++it;
    HPX_TEST_EQ(*it, 42);
    ++it;
    HPX_TEST(it == end);

    HPX_TEST_EQ(std::ranges::distance(result.begin(), result.end()), 3);
}

void test_repeat_infinite()
{
    auto result = hpx::views::repeat(10);

    auto it = result.begin();

    for (int i = 0; i < 100; ++i)
    {
        HPX_TEST_EQ(*it, 10);
        ++it;
    }
}

int main(int, char*[])
{
    test_repeat_finite();
    test_repeat_infinite();

    return hpx::util::report_errors();
}

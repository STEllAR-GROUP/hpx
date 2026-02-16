//  Copyright (c) 2026 Arpit Khandelwal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/util/slide.hpp>

#include <iterator>
#include <vector>

void test_slide()
{
    std::vector<int> v = {1, 2, 3, 4, 5};
    // slide(3) -> {1,2,3}, {2,3,4}, {3,4,5}
    auto result = hpx::views::slide(v, 3);

    int count = 0;
    auto it = result.begin();
    auto end = result.end();

    HPX_TEST_EQ(it != end, true);

    auto w1 = *it;
    HPX_TEST_EQ(std::distance(w1.begin(), w1.end()), 3);
    HPX_TEST_EQ(*w1.begin(), 1);

    ++it;
    auto w2 = *it;
    HPX_TEST_EQ(std::distance(w2.begin(), w2.end()), 3);
    HPX_TEST_EQ(*w2.begin(), 2);

    ++it;
    auto w3 = *it;
    HPX_TEST_EQ(std::distance(w3.begin(), w3.end()), 3);
    HPX_TEST_EQ(*w3.begin(), 3);

    ++it;
    HPX_TEST(it == end);
}

void test_slide_small()
{
    // size < n
    std::vector<int> v = {1, 2};
    auto result = hpx::views::slide(v, 3);
    HPX_TEST(result.begin() == result.end());
}

void test_slide_exact()
{
    // size == n
    std::vector<int> v = {1, 2, 3};
    auto result = hpx::views::slide(v, 3);
    auto it = result.begin();
    HPX_TEST(it != result.end());
    auto w1 = *it;
    HPX_TEST_EQ(std::distance(w1.begin(), w1.end()), 3);

    ++it;
    HPX_TEST(it == result.end());
}

int main(int, char*[])
{
    test_slide();
    test_slide_small();
    test_slide_exact();

    return hpx::util::report_errors();
}

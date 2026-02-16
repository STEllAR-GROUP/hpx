//  Copyright (c) 2026 Arpit Khandelwal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/util/chunk_by.hpp>

#include <vector>

void test_chunk_by()
{
    std::vector<int> v = {1, 1, 2, 2, 3, 3};
    auto result = hpx::views::chunk_by(v, std::equal_to<int>{});

    int count = 0;
    for (auto subrange : result)
    {
        count++;
        HPX_TEST_EQ(subrange.begin() != subrange.end(), true);
    }
    HPX_TEST_EQ(count, 3);
}

void test_chunk_by_predicate()
{
    std::vector<int> v = {1, 2, 3, 2, 3, 4, 1};

    auto result = hpx::views::chunk_by(v, std::less_equal<int>{});

    auto it = result.begin();
    auto c1 = *it;
    HPX_TEST_EQ(std::distance(c1.begin(), c1.end()), 3);
    HPX_TEST_EQ(*c1.begin(), 1);

    ++it;
    auto c2 = *it;
    HPX_TEST_EQ(std::distance(c2.begin(), c2.end()), 3);
    HPX_TEST_EQ(*c2.begin(), 2);

    ++it;
    auto c3 = *it;
    HPX_TEST_EQ(std::distance(c3.begin(), c3.end()), 1);
    HPX_TEST_EQ(*c3.begin(), 1);
}

void test_chunk_by_empty()
{
    std::vector<int> v;
    auto result = hpx::views::chunk_by(v, std::equal_to<int>{});
    HPX_TEST(result.begin() == result.end());
}

int main(int, char*[])
{
    test_chunk_by();
    test_chunk_by_predicate();
    test_chunk_by_empty();

    return hpx::util::report_errors();
}

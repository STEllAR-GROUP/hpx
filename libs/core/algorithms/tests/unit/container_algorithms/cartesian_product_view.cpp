//  Copyright (c) 2026 Arpit Khandelwal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/util/cartesian_product.hpp>

#include <iterator>
#include <vector>

void test_cartesian_product()
{
    std::vector<int> v1 = {1, 2};
    std::vector<int> v2 = {3, 4};

    // cartesian_product(v1, v2) -> (1,3), (1,4), (2,3), (2,4)
    auto result = hpx::views::cartesian_product(v1, v2);

    auto it = result.begin();
    auto end = result.end();

    HPX_TEST_EQ(hpx::get<0>(*it), 1);
    HPX_TEST_EQ(hpx::get<1>(*it), 3);

    ++it;
    HPX_TEST_EQ(hpx::get<0>(*it), 1);
    HPX_TEST_EQ(hpx::get<1>(*it), 4);

    ++it;
    HPX_TEST_EQ(hpx::get<0>(*it), 2);
    HPX_TEST_EQ(hpx::get<1>(*it), 3);

    ++it;
    HPX_TEST_EQ(hpx::get<0>(*it), 2);
    HPX_TEST_EQ(hpx::get<1>(*it), 4);

    ++it;
    HPX_TEST(it == end);
}

void test_cartesian_product_empty()
{
    std::vector<int> v1 = {1};
    std::vector<int> v2 = {};

    auto result = hpx::views::cartesian_product(v1, v2);
    HPX_TEST(result.begin() == result.end());
}

int main(int, char*[])
{
    test_cartesian_product();
    test_cartesian_product_empty();

    return hpx::util::report_errors();
}

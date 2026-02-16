//  Copyright (c) 2026 Arpit Khandelwal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/container_algorithms/fold_left.hpp>

#include <numeric>
#include <string>
#include <vector>

void test_fold_left()
{
    std::vector<int> v = {1, 2, 3, 4, 5};
    int init = 0;
    auto result = hpx::ranges::fold_left(v, init, std::plus<int>{});
    HPX_TEST_EQ(result, 15);

    auto result_iter =
        hpx::ranges::fold_left(v.begin(), v.end(), init, std::plus<int>{});
    HPX_TEST_EQ(result_iter, 15);
}

void test_fold_left_empty()
{
    std::vector<int> v;
    int init = 100;
    auto result = hpx::ranges::fold_left(v, init, std::plus<int>{});
    HPX_TEST_EQ(result, 100);
}

void test_fold_left_with_iter()
{
    std::vector<int> v = {1, 2, 3, 4, 5};
    int init = 0;
    auto result = hpx::ranges::fold_left_with_iter(v, init, std::plus<int>{});
    HPX_TEST(result.in == v.end());
    HPX_TEST_EQ(result.value, 15);
}

void test_fold_left_first()
{
    std::vector<int> v = {1, 2, 3, 4, 5};
    auto result = hpx::ranges::fold_left_first(v, std::plus<int>{});
    HPX_TEST(result.has_value());
    HPX_TEST_EQ(*result, 15);
}

void test_fold_left_first_empty()
{
    std::vector<int> v;
    auto result = hpx::ranges::fold_left_first(v, std::plus<int>{});
    HPX_TEST(!result.has_value());
}

void test_fold_left_first_with_iter()
{
    std::vector<int> v = {1, 2, 3, 4, 5};
    auto result = hpx::ranges::fold_left_first_with_iter(v, std::plus<int>{});
    HPX_TEST(result.in == v.end());
    HPX_TEST(result.value.has_value());
    HPX_TEST_EQ(*result.value, 15);
}

int main(int argc, char* argv[])
{
    test_fold_left();
    test_fold_left_empty();
    test_fold_left_with_iter();
    test_fold_left_first();
    test_fold_left_first_empty();
    test_fold_left_first_with_iter();

    return hpx::util::report_errors();
}

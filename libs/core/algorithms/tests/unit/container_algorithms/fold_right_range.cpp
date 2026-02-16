//  Copyright (c) 2026 Arpit Khandelwal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/container_algorithms/fold_right.hpp>

#include <numeric>
#include <string>
#include <vector>

void test_fold_right()
{
    std::vector<int> v = {1, 2, 3, 4, 5};
    int init = 0;
    auto result = hpx::ranges::fold_right(v, init, std::plus<int>{});
    HPX_TEST_EQ(result, 15);

    auto result_iter =
        hpx::ranges::fold_right(v.begin(), v.end(), init, std::plus<int>{});
    HPX_TEST_EQ(result_iter, 15);
}

void test_fold_right_empty()
{
    std::vector<int> v;
    int init = 100;
    auto result = hpx::ranges::fold_right(v, init, std::plus<int>{});
    HPX_TEST_EQ(result, 100);
}

void test_fold_right_last()
{
    std::vector<int> v = {1, 2, 3, 4, 5};
    auto result = hpx::ranges::fold_right_last(v, std::plus<int>{});
    HPX_TEST(result.has_value());
    HPX_TEST_EQ(*result, 15);
}

void test_fold_right_last_empty()
{
    std::vector<int> v;
    auto result = hpx::ranges::fold_right_last(v, std::plus<int>{});
    HPX_TEST(!result.has_value());
}

// Test with non-commutative operation to ensure right fold order
void test_fold_right_order()
{
    // API: (element, accum) -> accum
    // fold_right( [a, b, c], init, f )
    // = f(a, f(b, f(c, init)))

    std::vector<std::string> v = {"a", "b", "c"};
    std::string init = "d";
    auto result = hpx::ranges::fold_right(
        v, init, [](std::string elem, std::string acc) { return elem + acc; });
    // "a" + ("b" + ("c" + "d")) = "abcd"
    HPX_TEST_EQ(result, std::string("abcd"));
}

int main(int argc, char* argv[])
{
    test_fold_right();
    test_fold_right_empty();
    test_fold_right_last();
    test_fold_right_last_empty();
    test_fold_right_order();

    return hpx::util::report_errors();
}

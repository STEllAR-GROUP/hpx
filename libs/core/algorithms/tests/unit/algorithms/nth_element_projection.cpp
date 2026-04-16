//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2017-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

struct S
{
    int val;
};

template <typename ExPolicy>
void test_nth_element_projection(ExPolicy policy)
{
    std::vector<S> v(100);
    for (int i = 0; i < 100; ++i)
        v[i].val = 100 - i;

    auto nth = v.begin() + 50;

    hpx::nth_element(policy, v.begin(), nth, v.end(), std::less<int>{}, &S::val);

    // After nth_element, the element at nth should be the 51st smallest element (which is 51)
    HPX_TEST_EQ(v[50].val, 51);

    // All elements before nth should be <= v[50].val
    for (auto it = v.begin(); it != nth; ++it)
    {
        HPX_TEST_LTE(it->val, v[50].val);
    }

    // All elements after nth should be >= v[50].val
    for (auto it = nth + 1; it != v.end(); ++it)
    {
        HPX_TEST(it->val >= v[50].val);
    }
}

template <typename ExPolicy>
void test_nth_element_projection_type_change(ExPolicy policy)
{
    std::vector<std::string> v = {"abc", "a", "abcd", "ab", "abcde"};
    auto nth = v.begin() + 2;

    // Sort by string length
    hpx::nth_element(policy, v.begin(), nth, v.end(), std::less<std::size_t>{},
        &std::string::length);

    HPX_TEST_EQ(v[2].length(), std::size_t(3));
}

int hpx_main()
{
    test_nth_element_projection(hpx::execution::seq);
    test_nth_element_projection(hpx::execution::par);
    test_nth_element_projection(hpx::execution::par_unseq);

    test_nth_element_projection_type_change(hpx::execution::seq);
    test_nth_element_projection_type_change(hpx::execution::par);
    test_nth_element_projection_type_change(hpx::execution::par_unseq);

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    hpx::local::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

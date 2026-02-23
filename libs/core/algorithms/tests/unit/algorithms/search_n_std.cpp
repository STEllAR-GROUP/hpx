//  Copyright (c) 2026 Arivoli Ramamoorthy
//  Distributed under the Boost Software License, Version 1.0.

#include <hpx/algorithm.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <vector>

void test_search_n_basic()
{
    std::vector<int> v = {1, 1, 1, 2, 1, 1, 1, 1};

    auto it_std = std::search_n(v.begin(), v.end(), 3, 1);
    auto it_hpx = hpx::search_n(v.begin(), v.end(), 3, 1);

    HPX_TEST(it_hpx == it_std);
}

void test_search_n_no_match()
{
    std::vector<int> v = {1, 2, 3, 4};

    auto it = hpx::search_n(v.begin(), v.end(), 2, 5);
    HPX_TEST(it == v.end());
}

void test_search_n_count_zero()
{
    std::vector<int> v = {1, 2, 3};

    auto it = hpx::search_n(v.begin(), v.end(), 0, 1);
    HPX_TEST(it == v.begin());
}

void test_search_n_count_too_large()
{
    std::vector<int> v = {1, 1};

    auto it = hpx::search_n(v.begin(), v.end(), 3, 1);
    HPX_TEST(it == v.end());
}

void test_search_n_at_end()
{
    std::vector<int> v = {2, 2, 1, 1, 1};

    auto it = hpx::search_n(v.begin(), v.end(), 3, 1);
    HPX_TEST(it == v.begin() + 2);
}

void test_search_n_predicate()
{
    std::vector<int> v = {2, 4, 6, 7, 8};

    auto pred = [](int a, int b) { return (a % 2) == (b % 2); };

    auto it_std = std::search_n(v.begin(), v.end(), 3, 0, pred);
    auto it_hpx = hpx::search_n(v.begin(), v.end(), 3, 0, pred);

    HPX_TEST(it_hpx == it_std);
}

void test_search_n_parallel()
{
    std::vector<int> v = {1, 1, 1, 2, 1, 1};

    auto it = hpx::search_n(hpx::execution::par, v.begin(), v.end(), 3, 1);

    HPX_TEST(it == v.begin());
}

int hpx_main()
{
    test_search_n_basic();
    test_search_n_no_match();
    test_search_n_count_zero();
    test_search_n_count_too_large();
    test_search_n_at_end();
    test_search_n_predicate();
    test_search_n_parallel();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");
    return hpx::util::report_errors();
}

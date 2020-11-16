//  Copyright (c) 2019 Piotr Mikolajczyk
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_set_operations.hpp>
#include <hpx/modules/testing.hpp>

void set_difference_small_test(int rounds)
{
    std::vector<int> set_a{1, 2, 3, 4, 5};
    std::vector<int> set_b{1, 2, 4};
    std::vector<int> a_minus_b(2);

    std::vector<int> perfect(2);
    std::set_difference(set_a.begin(), set_a.end(), set_b.begin(), set_b.end(),
        perfect.begin());

    while (--rounds)
    {
        hpx::set_difference(hpx::execution::par, set_a.begin(), set_a.end(),
            set_b.begin(), set_b.end(), a_minus_b.begin());
        HPX_TEST(perfect == a_minus_b);
    }
}

void set_difference_medium_test(int rounds)
{
    std::vector<int> set_a(50);
    std::vector<int> set_b(20);

    std::iota(set_a.begin(), set_a.end(), 1);
    std::iota(set_b.begin(), set_b.end(), 2);

    std::vector<int> a_minus_b(50);

    std::vector<int> perfect(50);
    std::set_difference(set_a.begin(), set_a.end(), set_b.begin(), set_b.end(),
        perfect.begin());

    while (--rounds)
    {
        hpx::set_difference(hpx::execution::par, set_a.begin(), set_a.end(),
            set_b.begin(), set_b.end(), a_minus_b.begin());
        HPX_TEST(perfect == a_minus_b);
    }
}

void set_difference_large_test(int rounds)
{
    std::vector<int> set_a(5000000);
    std::vector<int> set_b(3000000);

    std::iota(set_a.begin(), set_a.end(), 1);
    std::fill(set_b.begin(), set_b.begin() + 1000000, 1);
    std::iota(set_b.begin() + 1000000, set_b.end(), 2);

    std::vector<int> a_minus_b(5000000);

    std::vector<int> perfect(5000000);
    std::set_difference(set_a.begin(), set_a.end(), set_b.begin(), set_b.end(),
        perfect.begin());

    while (--rounds)
    {
        hpx::set_difference(hpx::execution::par, set_a.begin(), set_a.end(),
            set_b.begin(), set_b.end(), a_minus_b.begin());
        HPX_TEST(perfect == a_minus_b);
    }
}

void set_difference_test(int rounds)
{
    set_difference_small_test(rounds);
    set_difference_medium_test(rounds);
    set_difference_large_test(rounds);
}

void set_intersection_small_test(int rounds)
{
    std::vector<int> set_a{1, 2, 3, 4, 5};
    std::vector<int> set_b{1, 2, 7};
    std::vector<int> a_and_b(2);

    std::vector<int> perfect(2);
    std::set_intersection(set_a.begin(), set_a.end(), set_b.begin(),
        set_b.end(), perfect.begin());

    while (--rounds)
    {
        hpx::set_intersection(hpx::execution::par, set_a.begin(), set_a.end(),
            set_b.begin(), set_b.end(), a_and_b.begin());
        HPX_TEST(perfect == a_and_b);
    }
}

void set_intersection_medium_test(int rounds)
{
    std::vector<int> set_a(50);
    std::vector<int> set_b(20);

    std::iota(set_a.begin(), set_a.end(), 1);
    std::iota(set_b.begin(), set_b.end(), 2);

    std::vector<int> a_and_b(20);

    std::vector<int> perfect(20);
    std::set_intersection(set_a.begin(), set_a.end(), set_b.begin(),
        set_b.end(), perfect.begin());

    while (--rounds)
    {
        hpx::set_intersection(hpx::execution::par, set_a.begin(), set_a.end(),
            set_b.begin(), set_b.end(), a_and_b.begin());
        HPX_TEST(perfect == a_and_b);
    }
}

void set_intersection_large_test(int rounds)
{
    std::vector<int> set_a(5000000);
    std::vector<int> set_b(3000000);

    std::iota(set_a.begin(), set_a.end(), 1);
    std::fill(set_b.begin(), set_b.begin() + 1000000, 1);
    std::iota(set_b.begin() + 1000000, set_b.end(), 2);

    std::vector<int> a_and_b(3000000);

    std::vector<int> perfect(3000000);
    std::set_intersection(set_a.begin(), set_a.end(), set_b.begin(),
        set_b.end(), perfect.begin());

    while (--rounds)
    {
        hpx::set_intersection(hpx::execution::par, set_a.begin(), set_a.end(),
            set_b.begin(), set_b.end(), a_and_b.begin());
        HPX_TEST(perfect == a_and_b);
    }
}

void set_intersection_test(int rounds)
{
    set_intersection_small_test(rounds);
    set_intersection_medium_test(rounds);
    set_intersection_large_test(rounds);
}

int hpx_main()
{
    int rounds = 5;
    set_intersection_test(rounds);
    set_difference_test(rounds);
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    hpx::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, init_args), 0,
        "HPX main exted with non-zero status");

    return hpx::util::report_errors();
}

//  Copyright (c) 2019 Piotr Mikolajczyk
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#ifdef HPX_WITH_CXX17_STD_EXECUTION_POLICES
#include <execution>
#endif

unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);

// returns random integer in range (rangeMin, rangeMax]
struct RandomIntInRange
{
    int rangeMin, rangeMax;
    RandomIntInRange(int rangeMin, int rangeMax)
      : rangeMin(rangeMin)
      , rangeMax(rangeMax) {};
    int operator()()
    {
        return (static_cast<int>(gen()) % (rangeMax - rangeMin + 1)) + rangeMin;
    }
};

void set_difference_randomized(int rounds, int maxLen)
{
    while (rounds--)
    {
        std::size_t len_a = gen() % maxLen, len_b = gen() % maxLen;
        std::vector<int> set_a(len_a), set_b(len_b);

        int rangeMin = 0;
        // rangeMax is set to increase probability of common elements
        int rangeMax = static_cast<int>((std::min) (len_a, len_b) * 2);

#ifdef HPX_WITH_CXX17_STD_EXECUTION_POLICES
        std::generate(std::execution::par_unseq, set_a.begin(), set_a.end(),
            RandomIntInRange(rangeMin, rangeMax));
        std::generate(std::execution::par_unseq, set_b.begin(), set_b.end(),
            RandomIntInRange(rangeMin, rangeMax));
#else
        std::generate(
            set_a.begin(), set_a.end(), RandomIntInRange(rangeMin, rangeMax));
        std::generate(
            set_b.begin(), set_b.end(), RandomIntInRange(rangeMin, rangeMax));
#endif
        std::sort(set_a.begin(), set_a.end());
        std::sort(set_b.begin(), set_b.end());

        len_a = std::unique(set_a.begin(), set_a.end()) - set_a.begin();
        len_b = std::unique(set_b.begin(), set_b.end()) - set_b.begin();

        set_a.resize(len_a);
        set_b.resize(len_b);

        // rand always gives non negative values, rangeMin >= 0
        std::vector<int> perfect((std::max) (len_a, len_b), -1);
        std::vector<int> a_minus_b((std::max) (len_a, len_b), -1);

        std::set_difference(set_a.begin(), set_a.end(), set_b.begin(),
            set_b.end(), perfect.begin());

        hpx::set_difference(hpx::execution::par, set_a.begin(), set_a.end(),
            set_b.begin(), set_b.end(), a_minus_b.begin());
        HPX_TEST(perfect == a_minus_b);
    }
}

void set_difference_small_test(int rounds)
{
    set_difference_randomized(rounds, 1 << 3);
}

void set_difference_medium_test(int rounds)
{
    set_difference_randomized(rounds, 1 << 10);
}

void set_difference_large_test(int rounds)
{
    set_difference_randomized(rounds, 1 << 20);
}

void set_difference_test(int rounds)
{
    set_difference_small_test(rounds);
    set_difference_medium_test(rounds);
    set_difference_large_test(rounds);
}

void set_intersection_small_test1(int rounds)
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

void set_intersection_small_test2(int rounds)
{
    std::vector<int> set_a{-1, 1, 2, 3, 4, 5};
    std::vector<int> set_b{0, 1, 2, 3, 8, 10};
    std::vector<int> a_and_b(3);

    std::vector<int> perfect(3);
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
    set_intersection_small_test1(rounds);
    set_intersection_small_test2(rounds);
    set_intersection_medium_test(rounds);
    set_intersection_large_test(rounds);
}

int hpx_main()
{
    int rounds = 5;
    set_intersection_test(rounds);
    set_difference_test(rounds);
    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    hpx::local::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exted with non-zero status");

    return hpx::util::report_errors();
}

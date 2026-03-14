//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2015 Daniel Bourgeois
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_sorted3(ExPolicy policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c_beg(10007);
    std::vector<int> c_end(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c_beg), std::end(c_beg), 0);
    std::iota(std::begin(c_end), std::end(c_end), 0);
    //add unsorted element to c_beg, c_end at the beginning, end respectively
    c_beg[0] = 20000;
    c_end[c_end.size() - 1] = -20000;

    bool is_ordered1 = hpx::ranges::is_sorted(
        policy, iterator(std::begin(c_beg)), iterator(std::end(c_beg)));
    bool is_ordered2 = hpx::ranges::is_sorted(
        policy, iterator(std::begin(c_end)), iterator(std::end(c_end)));
    bool is_ordered3 = hpx::ranges::is_sorted(policy,
        iterator(std::begin(c_beg)), iterator(std::end(c_beg)),
        std::less<int>(), [](int x) { return x == 20000 ? 0 : x; });
    bool is_ordered4 = hpx::ranges::is_sorted(policy,
        iterator(std::begin(c_end)), iterator(std::end(c_end)),
        std::less<int>(), [](int x) { return x == -20000 ? 10006 : x; });

    HPX_TEST(!is_ordered1);
    HPX_TEST(!is_ordered2);
    HPX_TEST(is_ordered3);
    HPX_TEST(is_ordered4);
}

template <typename ExPolicy, typename IteratorTag>
void test_sorted3_async(ExPolicy p, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c_beg(10007);
    std::vector<int> c_end(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c_beg), std::end(c_beg), 0);
    std::iota(std::begin(c_end), std::end(c_end), 0);
    //add unsorted element to c_beg, c_end at the beginning, end respectively
    c_beg[0] = 20000;
    c_end[c_end.size() - 1] = -20000;

    hpx::future<bool> f1 = hpx::ranges::is_sorted(
        p, iterator(std::begin(c_beg)), iterator(std::end(c_beg)));
    hpx::future<bool> f2 = hpx::ranges::is_sorted(
        p, iterator(std::begin(c_end)), iterator(std::end(c_end)));
    hpx::future<bool> f3 = hpx::ranges::is_sorted(p,
        iterator(std::begin(c_beg)), iterator(std::end(c_beg)),
        std::less<int>(), [](int x) { return x == 20000 ? 0 : x; });
    hpx::future<bool> f4 = hpx::ranges::is_sorted(p,
        iterator(std::begin(c_end)), iterator(std::end(c_end)),
        std::less<int>(), [](int x) { return x == -20000 ? 10006 : x; });
    f1.wait();
    HPX_TEST(!f1.get());
    f2.wait();
    HPX_TEST(!f2.get());
    f3.wait();
    HPX_TEST(f3.get());
    f4.wait();
    HPX_TEST(f4.get());
}

template <typename IteratorTag>
void test_sorted3_seq(IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c_beg(10007);
    std::vector<int> c_end(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c_beg), std::end(c_beg), 0);
    std::iota(std::begin(c_end), std::end(c_end), 0);
    //add unsorted element to c_beg, c_end at the beginning, end respectively
    c_beg[0] = 20000;
    c_end[c_end.size() - 1] = -20000;

    HPX_TEST(!hpx::ranges::is_sorted(
        iterator(std::begin(c_beg)), iterator(std::end(c_beg))));
    HPX_TEST(!hpx::ranges::is_sorted(
        iterator(std::begin(c_end)), iterator(std::end(c_end))));
    HPX_TEST(hpx::ranges::is_sorted(iterator(std::begin(c_beg)),
        iterator(std::end(c_beg)), std::less<int>(),
        [](int x) { return x == 20000 ? 0 : x; }));
    HPX_TEST(hpx::ranges::is_sorted(iterator(std::begin(c_end)),
        iterator(std::end(c_end)), std::less<int>(),
        [](int x) { return x == -20000 ? 10006 : x; }));
}

template <typename ExPolicy>
void test_sorted3(ExPolicy policy)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    std::vector<int> c_beg(10007);
    std::vector<int> c_end(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c_beg), std::end(c_beg), 0);
    std::iota(std::begin(c_end), std::end(c_end), 0);
    //add unsorted element to c_beg, c_end at the beginning, end respectively
    c_beg[0] = 20000;
    c_end[c_end.size() - 1] = -20000;

    HPX_TEST(!hpx::ranges::is_sorted(policy, c_beg));
    HPX_TEST(!hpx::ranges::is_sorted(policy, c_end));
    HPX_TEST(hpx::ranges::is_sorted(policy, c_beg, std::less<int>(),
        [](int x) { return x == 20000 ? 0 : x; }));
    HPX_TEST(hpx::ranges::is_sorted(policy, c_end, std::less<int>(),
        [](int x) { return x == -20000 ? 10006 : x; }));
}

template <typename ExPolicy>
void test_sorted3_async(ExPolicy p)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    std::vector<int> c_beg(10007);
    std::vector<int> c_end(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c_beg), std::end(c_beg), 0);
    std::iota(std::begin(c_end), std::end(c_end), 0);
    //add unsorted element to c_beg, c_end at the beginning, end respectively
    c_beg[0] = 20000;
    c_end[c_end.size() - 1] = -20000;

    hpx::future<bool> f1 = hpx::ranges::is_sorted(p, c_beg);
    hpx::future<bool> f2 = hpx::ranges::is_sorted(p, c_end);
    hpx::future<bool> f3 = hpx::ranges::is_sorted(
        p, c_beg, std::less<int>(), [](int x) { return x == 20000 ? 0 : x; });
    hpx::future<bool> f4 = hpx::ranges::is_sorted(p, c_end, std::less<int>(),
        [](int x) { return x == -20000 ? 10006 : x; });

    f1.wait();
    HPX_TEST(!f1.get());
    f2.wait();
    HPX_TEST(!f2.get());
    f3.wait();
    HPX_TEST(f3.get());
    f4.wait();
    HPX_TEST(f4.get());
}

void test_sorted3_seq()
{
    std::vector<int> c_beg(10007);
    std::vector<int> c_end(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c_beg), std::end(c_beg), 0);
    std::iota(std::begin(c_end), std::end(c_end), 0);
    //add unsorted element to c_beg, c_end at the beginning, end respectively
    c_beg[0] = 20000;
    c_end[c_end.size() - 1] = -20000;

    HPX_TEST(!hpx::ranges::is_sorted(c_beg));
    HPX_TEST(!hpx::ranges::is_sorted(c_end));
    HPX_TEST(hpx::ranges::is_sorted(
        c_beg, std::less<int>(), [](int x) { return x == 20000 ? 0 : x; }));
    HPX_TEST(hpx::ranges::is_sorted(c_end, std::less<int>(),
        [](int x) { return x == -20000 ? 10006 : x; }));
}

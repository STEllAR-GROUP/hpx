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

using identity = hpx::identity;

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_sorted2(ExPolicy policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c), std::end(c), 0);
    //Add a certain large value in middle of array to ignore
    int ignore = 20000;
    c[c.size() / 2] = ignore;
    //Provide custom predicate to ignore the value of ignore
    //pred should return true when it is given something deemed not sorted
    auto pred = [&ignore](int ahead, int behind) {
        return behind > ahead && behind != ignore;
    };

    HPX_TEST(hpx::ranges::is_sorted(policy, iterator(std::begin(c)),
        iterator(std::end(c)), pred, identity()));
    HPX_TEST(!hpx::ranges::is_sorted(policy, iterator(std::begin(c)),
        iterator(std::end(c)), pred,
        [ignore](int x) { return x == ignore ? -x : x; }));
}

template <typename ExPolicy, typename IteratorTag>
void test_sorted2_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c), std::end(c), 0);
    //Add a certain large value in middle of array to ignore
    int ignore = 20000;
    c[c.size() / 2] = ignore;
    //Provide custom predicate to ignore the value of ignore
    //pred should return true when it is given something deemed not sorted
    auto pred = [&ignore](int ahead, int behind) {
        return behind > ahead && behind != ignore;
    };

    hpx::future<bool> f1 = hpx::ranges::is_sorted(
        p, iterator(std::begin(c)), iterator(std::end(c)), pred);
    f1.wait();
    HPX_TEST(f1.get());
    hpx::future<bool> f2 = hpx::ranges::is_sorted(p, iterator(std::begin(c)),
        iterator(std::end(c)), pred,
        [ignore](int x) { return x == ignore ? -x : x; });
    f2.wait();
    HPX_TEST(!f2.get());
}

template <typename IteratorTag>
void test_sorted2_seq(IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c), std::end(c), 0);
    //Add a certain large value in middle of array to ignore
    int ignore = 20000;
    c[c.size() / 2] = ignore;
    //Provide custom predicate to ignore the value of ignore
    //pred should return true when it is given something deemed not sorted
    auto pred = [&ignore](int ahead, int behind) {
        return behind > ahead && behind != ignore;
    };

    HPX_TEST(hpx::ranges::is_sorted(
        iterator(std::begin(c)), iterator(std::end(c)), pred));
    HPX_TEST(
        !hpx::ranges::is_sorted(iterator(std::begin(c)), iterator(std::end(c)),
            pred, [ignore](int x) { return x == ignore ? -x : x; }));
}

template <typename ExPolicy>
void test_sorted2(ExPolicy policy)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    std::vector<int> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c), std::end(c), 0);
    //Add a certain large value in middle of array to ignore
    int ignore = 20000;
    c[c.size() / 2] = ignore;
    //Provide custom predicate to ignore the value of ignore
    //pred should return true when it is given something deemed not sorted
    auto pred = [&ignore](int ahead, int behind) {
        return behind > ahead && behind != ignore;
    };

    HPX_TEST(hpx::ranges::is_sorted(policy, c, pred));
    HPX_TEST(!hpx::ranges::is_sorted(
        policy, c, pred, [ignore](int x) { return x == ignore ? -x : x; }));
}

template <typename ExPolicy>
void test_sorted2_async(ExPolicy p)
{
    std::vector<int> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c), std::end(c), 0);
    //Add a certain large value in middle of array to ignore
    int ignore = 20000;
    c[c.size() / 2] = ignore;
    //Provide custom predicate to ignore the value of ignore
    //pred should return true when it is given something deemed not sorted
    auto pred = [&ignore](int ahead, int behind) {
        return behind > ahead && behind != ignore;
    };

    hpx::future<bool> f1 = hpx::ranges::is_sorted(p, c, pred);
    f1.wait();
    HPX_TEST(f1.get());
    hpx::future<bool> f2 = hpx::ranges::is_sorted(
        p, c, pred, [ignore](int x) { return x == ignore ? -x : x; });
    f2.wait();
    HPX_TEST(!f2.get());
}

void test_sorted2_seq()
{
    std::vector<int> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c), std::end(c), 0);
    //Add a certain large value in middle of array to ignore
    int ignore = 20000;
    c[c.size() / 2] = ignore;
    //Provide custom predicate to ignore the value of ignore
    //pred should return true when it is given something deemed not sorted
    auto pred = [&ignore](int ahead, int behind) {
        return behind > ahead && behind != ignore;
    };

    HPX_TEST(hpx::ranges::is_sorted(c, pred));
    HPX_TEST(!hpx::ranges::is_sorted(
        c, pred, [ignore](int x) { return x == ignore ? -x : x; }));
}

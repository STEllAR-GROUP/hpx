//  Copyright (c) 2014-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/modules/testing.hpp>
#include <hpx/parallel/algorithms/fold.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "test_utils.hpp"

int seed = std::random_device{}();
std::mt19937 gen(seed);

// fold_left(begin, end, init, op)
template <typename IteratorTag>
void fold_left_test1(IteratorTag)
{
    using base_iterator = std::vector<int>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<int> c = {1, 2, 3, 4, 5};

    int val(1);
    auto op = [](auto v1, auto v2) { return v1 * v2; };

    int r1 =
        hpx::fold_left(iterator(std::begin(c)), iterator(std::end(c)), val, op);

    // verify values
    int r2 = 120;
    HPX_TEST_EQ(r1, r2);
}

// fold_left(policy, begin, end, init, op)
template <typename ExPolicy, typename IteratorTag>
void fold_left_test1(ExPolicy policy, IteratorTag)
{
    using base_iterator = std::vector<int>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<int> c = {1, 2, 3, 4, 5};

    int val(1);
    auto op = [](auto v1, auto v2) { return v1 * v2; };

    int r1 = hpx::fold_left(
        policy, iterator(std::begin(c)), iterator(std::end(c)), val, op);

    // verify values
    int r2 = 120;
    HPX_TEST_EQ(r1, r2);
}

// fold_right(begin, end, init, op)
template <typename IteratorTag>
void fold_right_test1(IteratorTag)
{
    using base_iterator = std::vector<int>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<int> c = {1, 2, 3, 4, 5};

    int val(1);
    auto op = [](auto v1, auto v2) { return v1 * v2; };

    int r1 =
        hpx::fold_right(iterator(std::begin(c)), iterator(std::end(c)), val, op);

    // verify values
    int r2 = 120;
    HPX_TEST_EQ(r1, r2);
}

// fold_right(policy, begin, end, init, op)
template <typename ExPolicy, typename IteratorTag>
void fold_right_test1(ExPolicy policy, IteratorTag)
{
    using base_iterator = std::vector<int>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<int> c = {1, 2, 3, 4, 5};

    int val(1);
    auto op = [](auto v1, auto v2) { return v1 * v2; };

    int r1 = hpx::fold_right(
        policy, iterator(std::begin(c)), iterator(std::end(c)), val, op);

    // verify values
    int r2 = 120;
    HPX_TEST_EQ(r1, r2);
}

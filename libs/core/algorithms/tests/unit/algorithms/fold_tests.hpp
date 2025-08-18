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

    int r1 = hpx::fold_right(
        iterator(std::begin(c)), iterator(std::end(c)), val, op);

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

// fold_left(begin, end, init, op)
template <typename IteratorTag>
void fold_left_first_test1(IteratorTag)
{
    using base_iterator = std::vector<int>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<int> c = {1, 2, 3, 4, 5};

    auto op = [](auto v1, auto v2) { return v1 * v2; };

    hpx::optional<int> r1 = hpx::fold_left_first(
        iterator(std::begin(c)), iterator(std::end(c)), op);

    // verify values
    int r2 = 120;
    HPX_TEST_EQ(r1.value(), r2);
}

// fold_left_first(policy, begin, end, op)
template <typename ExPolicy, typename IteratorTag>
void fold_left_first_test1(ExPolicy policy, IteratorTag)
{
    using base_iterator = std::vector<int>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<int> c = {1, 2, 3, 4, 5};

    auto op = [](auto v1, auto v2) { return v1 * v2; };

    hpx::optional<int> r1 = hpx::fold_left_first(
        policy, iterator(std::begin(c)), iterator(std::end(c)), op);

    // verify values
    int r2 = 120;
    HPX_TEST_EQ(r1.value(), r2);
}

// fold_right_first(begin, end, op)
template <typename IteratorTag>
void fold_right_first_test1(IteratorTag)
{
    using base_iterator = std::vector<int>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<int> c = {1, 2, 3, 4, 5};

    auto op = [](auto v1, auto v2) { return v1 * v2; };

    hpx::optional<int> r1 = hpx::fold_right_first(
        iterator(std::begin(c)), iterator(std::end(c)), op);

    // verify values
    int r2 = 120;
    HPX_TEST_EQ(r1.value(), r2);
}

// fold_right_first(policy, begin, end, op)
template <typename ExPolicy, typename IteratorTag>
void fold_right_first_test1(ExPolicy policy, IteratorTag)
{
    using base_iterator = std::vector<int>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<int> c = {1, 2, 3, 4, 5};

    auto op = [](auto v1, auto v2) { return v1 * v2; };

    hpx::optional<int> r1 = hpx::fold_right_first(
        policy, iterator(std::begin(c)), iterator(std::end(c)), op);

    // verify values
    int r2 = 120;
    HPX_TEST_EQ(r1.value(), r2);
}

// fold_left(begin, end, init, op)
template <typename IteratorTag>
void fold_left_with_iter_test1(IteratorTag)
{
    using base_iterator = std::vector<int>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<int> c = {1, 2, 3, 4, 5};

    int val(1);
    auto op = [](auto v1, auto v2) { return v1 * v2; };

    hpx::ranges::in_value_result<iterator, int> r1 = hpx::fold_left_with_iter(
        iterator(std::begin(c)), iterator(std::end(c)), val, op);

    // verify values
    int r2 = 120;
    HPX_TEST_EQ(r1.value, r2);
    HPX_TEST(r1.in == iterator(std::end(c)));
}

// fold_left_first_with_iter(policy, begin, end, init, op)
template <typename ExPolicy, typename IteratorTag>
void fold_left_with_iter_test1(ExPolicy policy, IteratorTag)
{
    using base_iterator = std::vector<int>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<int> c = {1, 2, 3, 4, 5};

    int val(1);
    auto op = [](auto v1, auto v2) { return v1 * v2; };

    hpx::ranges::in_value_result<iterator, hpx::optional<int>> r1 =
        hpx::fold_left_with_iter(
            policy, iterator(std::begin(c)), iterator(std::end(c)), val, op);

    // verify values
    int r2 = 120;
    HPX_TEST_EQ(r1.value.value(), r2);
    HPX_TEST(r1.in == iterator(std::end(c)));
}

// fold_left(begin, end, init, op)
template <typename IteratorTag>
void fold_left_first_with_iter_test1(IteratorTag)
{
    using base_iterator = std::vector<int>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<int> c = {1, 2, 3, 4, 5};

    auto op = [](auto v1, auto v2) { return v1 * v2; };

    hpx::ranges::in_value_result<iterator, hpx::optional<int>> r1 =
        hpx::fold_left_first_with_iter(
            iterator(std::begin(c)), iterator(std::end(c)), op);

    // verify values
    int r2 = 120;
    HPX_TEST_EQ(r1.value.value(), r2);
    HPX_TEST(r1.in == iterator(std::end(c)));
}

// fold_left_first_with_iter(policy, begin, end, init, op)
template <typename ExPolicy, typename IteratorTag>
void fold_left_first_with_iter_test1(ExPolicy policy, IteratorTag)
{
    using base_iterator = std::vector<int>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<int> c = {1, 2, 3, 4, 5};

    auto op = [](auto v1, auto v2) { return v1 * v2; };

    hpx::ranges::in_value_result<iterator, hpx::optional<int>> r1 =
        hpx::fold_left_first_with_iter(
            policy, iterator(std::begin(c)), iterator(std::end(c)), op);

    // verify values
    int r2 = 120;
    HPX_TEST_EQ(r1.value.value(), r2);
    HPX_TEST(r1.in == iterator(std::end(c)));
}

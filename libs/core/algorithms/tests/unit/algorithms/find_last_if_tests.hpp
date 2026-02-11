//  Copyright (c) 2026 Arpit Khandelwal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>

#include "test_utils.hpp"

template <typename IteratorTag>
void test_find_last_if()
{
    using namespace hpx::execution;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand() + 1);
    c[c.size() / 2] = 1;

    // test sequential
    {
        auto result = hpx::find_last_if(
            test::test_iterator<std::vector<int>::iterator, IteratorTag>(
                std::begin(c)),
            test::test_iterator<std::vector<int>::iterator, IteratorTag>(
                std::end(c)),
            [](int v) { return v == 1; });

        HPX_TEST_EQ(*result, 1);
    }
    {
        auto result = hpx::find_last_if(seq,
            test::test_iterator<std::vector<int>::iterator, IteratorTag>(
                std::begin(c)),
            test::test_iterator<std::vector<int>::iterator, IteratorTag>(
                std::end(c)),
            [](int v) { return v == 1; });

        HPX_TEST_EQ(*result, 1);
    }

    // test parallel
    {
        auto result = hpx::find_last_if(par,
            test::test_iterator<std::vector<int>::iterator, IteratorTag>(
                std::begin(c)),
            test::test_iterator<std::vector<int>::iterator, IteratorTag>(
                std::end(c)),
            [](int v) { return v == 1; });

        HPX_TEST_EQ(*result, 1);
    }
}

template <typename IteratorTag>
void test_find_last_if_exception()
{
    // Implementation for exception testing
}

template <typename IteratorTag>
void test_find_last_if_bad_alloc()
{
    // Implementation for bad_alloc testing
}

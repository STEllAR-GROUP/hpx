//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2020 Hartmut Kaiser
//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/algorithm.hpp>
#include <hpx/datapar.hpp>
#include <hpx/execution.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "../algorithms/test_utils.hpp"

struct foo
{
    template <typename T>
    T operator()()
    {
        return T(10);
    }
};

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_generate(ExPolicy&& policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);

    foo gen;

    hpx::generate(policy, iterator(std::begin(c)), iterator(std::end(c)), gen);

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](int v) -> void {
        HPX_TEST_EQ(v, int(10));
        ++count;
    });
    HPX_TEST_EQ(count, c.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_generate_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);

    foo gen;

    hpx::future<void> f =
        hpx::generate(p, iterator(std::begin(c)), iterator(std::end(c)), gen);
    f.wait();

    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](int v) -> void {
        HPX_TEST_EQ(v, int(10));
        ++count;
    });
    HPX_TEST_EQ(count, c.size());
}

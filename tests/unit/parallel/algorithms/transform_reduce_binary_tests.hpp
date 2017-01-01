//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TEST_TRANSFORM_REDUCE_BINARY_SEP_13_2016_1227PM)
#define HPX_PARALLEL_TEST_TRANSFORM_REDUCE_BINARY_SEP_13_2016_1227PM

#include <hpx/include/parallel_transform_reduce.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_reduce_binary(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c = test::random_iota<int>(1007);
    std::vector<int> d = test::random_iota<int>(1007);
    int init = std::rand() % 1007; //-V101

    int r = hpx::parallel::transform_reduce(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)),
        boost::begin(d), init);

    HPX_TEST_EQ(r, std::inner_product(
        boost::begin(c), boost::end(c), boost::begin(d), init));
}

template <typename ExPolicy, typename IteratorTag>
void test_transform_reduce_binary_async(ExPolicy p, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c = test::random_iota<int>(1007);
    std::vector<int> d = test::random_iota<int>(1007);
    int init = std::rand() % 1007; //-V101

    hpx::future<int> fut_r =
        hpx::parallel::transform_reduce(p, iterator(boost::begin(c)),
        iterator(boost::end(c)), boost::begin(d), init);

    fut_r.wait();
    HPX_TEST_EQ(fut_r.get(), std::inner_product(
        boost::begin(c), boost::end(c), boost::begin(d), init));
}

#endif

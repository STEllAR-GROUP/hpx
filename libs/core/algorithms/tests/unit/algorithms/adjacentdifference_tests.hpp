//  Copyright (c) 2015 Daniel Bourgeois
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/modules/testing.hpp>
#include <hpx/parallel/algorithms/adjacent_difference.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_adjacent_difference(ExPolicy policy)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    std::vector<int> c = test::random_iota<int>(10007);
    std::vector<int> d(10007);
    std::vector<int> d_ans(10007);

    auto it = hpx::parallel::adjacent_difference(
        policy, std::begin(c), std::end(c), std::begin(d));
    std::adjacent_difference(std::begin(c), std::end(c), std::begin(d_ans));

    HPX_TEST(std::equal(std::begin(d), std::end(d), std::begin(d_ans),
        [](auto lhs, auto rhs) { return lhs == rhs; }));

    HPX_TEST(std::end(d) == it);
}

template <typename ExPolicy>
void test_adjacent_difference_async(ExPolicy p)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    std::vector<int> c = test::random_iota<int>(10007);
    std::vector<int> d(10007);
    std::vector<int> d_ans(10007);

    auto f_it = hpx::parallel::adjacent_difference(
        p, std::begin(c), std::end(c), std::begin(d));
    std::adjacent_difference(std::begin(c), std::end(c), std::begin(d_ans));

    f_it.wait();
    HPX_TEST(std::equal(std::begin(d), std::end(d), std::begin(d_ans),
        [](auto lhs, auto rhs) { return lhs == rhs; }));

    HPX_TEST(std::end(d) == f_it.get());
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_adjacent_difference_exception(ExPolicy policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;
    std::vector<int> c(10007);
    std::vector<int> d(10007);

    bool caught_exception = false;
    try
    {
        hpx::parallel::adjacent_difference(policy,
            decorated_iterator(std::begin(c)), decorated_iterator(std::end(c)),
            std::begin(d), [](auto lhs, auto rhs) {
                throw std::runtime_error("test");
                return lhs - rhs;
            });
        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(policy, e);
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_adjacent_difference_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<int> c(10007);
    std::vector<int> d(10007);

    bool caught_exception = false;
    bool returned_from_algorithm = false;

    try
    {
        hpx::future<base_iterator> f = hpx::parallel::adjacent_difference(p,
            decorated_iterator(std::begin(c)), decorated_iterator(std::end(c)),
            std::begin(d), [](auto lhs, auto rhs) {
                throw std::runtime_error("test");
                return lhs - rhs;
            });

        returned_from_algorithm = true;

        f.get();

        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(p, e);
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST(returned_from_algorithm);
}

//////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_adjacent_difference_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<int> c(10007);
    std::vector<int> d(10007);

    bool caught_bad_alloc = false;
    try
    {
        hpx::parallel::adjacent_difference(policy,
            decorated_iterator(std::begin(c)), decorated_iterator(std::end(c)),
            std::begin(d), [](auto lhs, auto rhs) {
                throw std::bad_alloc();
                return lhs - rhs;
            });
    }
    catch (std::bad_alloc const&)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
}

template <typename ExPolicy, typename IteratorTag>
void test_adjacent_difference_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<int> c(10007);
    std::vector<int> d(10007);

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;

    try
    {
        hpx::future<base_iterator> f = hpx::parallel::adjacent_difference(p,
            decorated_iterator(std::begin(c)), decorated_iterator(std::end(c)),
            std::begin(d), [](auto lhs, auto rhs) {
                throw std::bad_alloc();
                return lhs - rhs;
            });
        returned_from_algorithm = true;

        f.get();

        HPX_TEST(false);
    }
    catch (std::bad_alloc const&)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
    HPX_TEST(returned_from_algorithm);
}

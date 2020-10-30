//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/include/parallel_container_algorithm.hpp>
#include <hpx/iterator_support/iterator_range.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////////
struct counter
{
    std::size_t count = 0;
    void operator()(std::size_t& v)
    {
        ++count;
        v = 42;
    }
};

template <typename IteratorTag>
void test_for_each_seq(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    counter f;
    auto res = hpx::ranges::for_each(
        hpx::util::make_iterator_range(
            iterator(std::begin(c)), iterator(std::end(c))),
        f);

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](std::size_t v) -> void {
        HPX_TEST_EQ(v, std::size_t(42));
        ++count;
    });
    HPX_TEST_EQ(count, c.size());
    HPX_TEST(res.in == iterator(std::end(c)));
    HPX_TEST_EQ(res.fun.count, c.size());
    HPX_TEST_EQ(f.count, c.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_for_each(ExPolicy&& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    iterator result = hpx::ranges::for_each(std::forward<ExPolicy>(policy),
        hpx::util::make_iterator_range(
            iterator(std::begin(c)), iterator(std::end(c))),
        [](std::size_t& v) { v = 42; });
    HPX_TEST(result == iterator(std::end(c)));

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](std::size_t v) -> void {
        HPX_TEST_EQ(v, std::size_t(42));
        ++count;
    });
    HPX_TEST_EQ(count, c.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_for_each_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    hpx::future<iterator> f = hpx::ranges::for_each(std::forward<ExPolicy>(p),
        hpx::util::make_iterator_range(
            iterator(std::begin(c)), iterator(std::end(c))),
        [](std::size_t& v) { v = 42; });
    HPX_TEST(f.get() == iterator(std::end(c)));

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](std::size_t v) -> void {
        HPX_TEST_EQ(v, std::size_t(42));
        ++count;
    });
    HPX_TEST_EQ(count, c.size());
}

////////////////////////////////////////////////////////////////////////////////
struct counter_exception
{
    std::size_t count = 0;
    void operator()(std::size_t&)
    {
        ++count;
        throw std::runtime_error("test");
    }
};

template <typename IteratorTag>
void test_for_each_exception_seq(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    counter_exception f;
    try
    {
        hpx::ranges::for_each(
            hpx::util::make_iterator_range(
                iterator(std::begin(c)), iterator(std::end(c))),
            f);

        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e)
    {
        caught_exception = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST_EQ(f.count, std::size_t(1));
}

template <typename ExPolicy, typename IteratorTag>
void test_for_each_exception(ExPolicy policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    try
    {
        hpx::ranges::for_each(policy,
            hpx::util::make_iterator_range(
                iterator(std::begin(c)), iterator(std::end(c))),
            [](std::size_t&) { throw std::runtime_error("test"); });

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
void test_for_each_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        hpx::future<iterator> f = hpx::ranges::for_each(p,
            hpx::util::make_iterator_range(
                iterator(std::begin(c)), iterator(std::end(c))),
            [](std::size_t&) { throw std::runtime_error("test"); });
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

////////////////////////////////////////////////////////////////////////////////
struct counter_bad_alloc
{
    std::size_t count = 0;
    void operator()(std::size_t&)
    {
        ++count;
        throw std::bad_alloc();
    }
};

template <typename IteratorTag>
void test_for_each_bad_alloc_seq(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    counter_bad_alloc f;
    try
    {
        hpx::ranges::for_each(
            hpx::util::make_iterator_range(
                iterator(std::begin(c)), iterator(std::end(c))),
            f);

        HPX_TEST(false);
    }
    catch (std::bad_alloc const&)
    {
        caught_exception = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST_EQ(f.count, std::size_t(1));
}

template <typename ExPolicy, typename IteratorTag>
void test_for_each_bad_alloc(ExPolicy policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    try
    {
        hpx::ranges::for_each(policy,
            hpx::util::make_iterator_range(
                iterator(std::begin(c)), iterator(std::end(c))),
            [](std::size_t&) { throw std::bad_alloc(); });

        HPX_TEST(false);
    }
    catch (std::bad_alloc const&)
    {
        caught_exception = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_for_each_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        hpx::future<iterator> f = hpx::ranges::for_each(p,
            hpx::util::make_iterator_range(
                iterator(std::begin(c)), iterator(std::end(c))),
            [](std::size_t&) { throw std::bad_alloc(); });
        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch (std::bad_alloc const&)
    {
        caught_exception = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST(returned_from_algorithm);
}

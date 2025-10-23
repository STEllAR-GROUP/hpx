//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/modules/iterator_support.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/container_algorithms/for_each.hpp>

#include <atomic>
#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////////
template <typename Proj>
struct counter
{
    Proj proj;
    std::size_t count = 0;
    void operator()(std::size_t v)
    {
        HPX_TEST_EQ(v, proj(static_cast<std::size_t>(42)));
        ++count;
    }
};

template <typename IteratorTag, typename Proj>
void test_for_each_seq(IteratorTag, Proj&& proj)
{
    using base_iterator = std::vector<std::size_t>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<std::size_t> c(10007);
    std::fill(std::begin(c), std::end(c), static_cast<std::size_t>(42));

    using counter_type = counter<std::decay_t<Proj>>;

    counter_type f{proj};
    auto res =
        hpx::ranges::for_each(hpx::util::iterator_range(iterator(std::begin(c)),
                                  iterator(std::end(c))),
            std::ref(f), proj);

    HPX_TEST(res.in == iterator(std::end(c)));
    HPX_TEST_EQ(static_cast<counter_type>(res.fun).count, c.size());
    HPX_TEST_EQ(f.count, c.size());
}

template <typename ExPolicy, typename IteratorTag, typename Proj>
void test_for_each(ExPolicy&& policy, IteratorTag, Proj&& proj)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value);

    using base_iterator = std::vector<std::size_t>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<std::size_t> c(10007);
    std::fill(std::begin(c), std::end(c), static_cast<std::size_t>(42));

    std::atomic<std::size_t> count(0);

    iterator result = hpx::ranges::for_each(
        std::forward<ExPolicy>(policy),
        hpx::util::iterator_range(
            iterator(std::begin(c)), iterator(std::end(c))),
        [&count, &proj](std::size_t v) {
            HPX_TEST_EQ(v, proj(static_cast<std::size_t>(42)));
            ++count;
        },
        proj);

    HPX_TEST(result == iterator(std::end(c)));
    HPX_TEST_EQ(count, c.size());
}

template <typename ExPolicy, typename IteratorTag, typename Proj>
void test_for_each_async(ExPolicy&& p, IteratorTag, Proj&& proj)
{
    using base_iterator = std::vector<std::size_t>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<std::size_t> c(10007);
    std::fill(std::begin(c), std::end(c), static_cast<std::size_t>(42));

    std::atomic<std::size_t> count(0);

    hpx::future<iterator> f = hpx::ranges::for_each(
        std::forward<ExPolicy>(p),
        hpx::util::iterator_range(
            iterator(std::begin(c)), iterator(std::end(c))),
        [&count, &proj](std::size_t v) {
            HPX_TEST_EQ(v, proj(static_cast<std::size_t>(42)));
            ++count;
        },
        proj);
    f.wait();

    HPX_TEST(f.get() == iterator(std::end(c)));
    HPX_TEST_EQ(count, c.size());
}

////////////////////////////////////////////////////////////////////////////////
struct counter_exception
{
    std::size_t count = 0;
    [[noreturn]] void operator()(std::size_t)
    {
        ++count;
        throw std::runtime_error("test");
    }
};

template <typename IteratorTag, typename Proj>
void test_for_each_exception_seq(IteratorTag, Proj&& proj)
{
    using base_iterator = std::vector<std::size_t>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<std::size_t> c(10007);
    std::fill(std::begin(c), std::end(c), static_cast<std::size_t>(42));

    bool caught_exception = false;
    counter_exception f;
    try
    {
        hpx::ranges::for_each(hpx::util::iterator_range(iterator(std::begin(c)),
                                  iterator(std::end(c))),
            std::ref(f), proj);

        HPX_TEST(false);
    }
    catch (hpx::exception_list const&)
    {
        caught_exception = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST_EQ(f.count, static_cast<std::size_t>(1));
}

template <typename ExPolicy, typename IteratorTag, typename Proj>
void test_for_each_exception(ExPolicy policy, IteratorTag, Proj&& proj)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value);

    using base_iterator = std::vector<std::size_t>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<std::size_t> c(10007);
    std::fill(std::begin(c), std::end(c), static_cast<std::size_t>(42));

    bool caught_exception = false;
    try
    {
        hpx::ranges::for_each(
            policy,
            hpx::util::iterator_range(
                iterator(std::begin(c)), iterator(std::end(c))),
            [](std::size_t) { throw std::runtime_error("test"); }, proj);

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

template <typename ExPolicy, typename IteratorTag, typename Proj>
void test_for_each_exception_async(ExPolicy p, IteratorTag, Proj&& proj)
{
    using base_iterator = std::vector<std::size_t>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<std::size_t> c(10007);
    std::fill(std::begin(c), std::end(c), static_cast<std::size_t>(42));

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        hpx::future<iterator> f = hpx::ranges::for_each(
            p,
            hpx::util::iterator_range(
                iterator(std::begin(c)), iterator(std::end(c))),
            [](std::size_t) { throw std::runtime_error("test"); }, proj);
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
    [[noreturn]] void operator()(std::size_t)
    {
        ++count;
        throw std::bad_alloc();
    }
};

template <typename IteratorTag, typename Proj>
void test_for_each_bad_alloc_seq(IteratorTag, Proj&& proj)
{
    using base_iterator = std::vector<std::size_t>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<std::size_t> c(10007);
    std::fill(std::begin(c), std::end(c), static_cast<std::size_t>(42));

    bool caught_exception = false;
    counter_bad_alloc f;
    try
    {
        hpx::ranges::for_each(hpx::util::iterator_range(iterator(std::begin(c)),
                                  iterator(std::end(c))),
            std::ref(f), proj);

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
    HPX_TEST_EQ(f.count, static_cast<std::size_t>(1));
}

template <typename ExPolicy, typename IteratorTag, typename Proj>
void test_for_each_bad_alloc(ExPolicy policy, IteratorTag, Proj&& proj)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value);

    using base_iterator = std::vector<std::size_t>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<std::size_t> c(10007);
    std::fill(std::begin(c), std::end(c), static_cast<std::size_t>(42));

    bool caught_exception = false;
    try
    {
        hpx::ranges::for_each(
            policy,
            hpx::util::iterator_range(
                iterator(std::begin(c)), iterator(std::end(c))),
            [](std::size_t) { throw std::bad_alloc(); }, proj);

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

template <typename ExPolicy, typename IteratorTag, typename Proj>
void test_for_each_bad_alloc_async(ExPolicy p, IteratorTag, Proj&& proj)
{
    using base_iterator = std::vector<std::size_t>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;

    std::vector<std::size_t> c(10007);
    std::fill(std::begin(c), std::end(c), static_cast<std::size_t>(42));

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        hpx::future<iterator> f = hpx::ranges::for_each(
            p,
            hpx::util::iterator_range(
                iterator(std::begin(c)), iterator(std::end(c))),
            [](std::size_t) { throw std::bad_alloc(); }, proj);
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

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
void test_sorted_exception(ExPolicy policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;
    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), 0);

    bool caught_exception = false;
    try
    {
        hpx::ranges::is_sorted(policy,
            decorated_iterator(
                std::begin(c), []() { throw std::runtime_error("test"); }),
            decorated_iterator(
                std::end(c), []() { throw std::runtime_error("test"); }));
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

    caught_exception = false;
    try
    {
        hpx::ranges::is_sorted(policy, iterator(std::begin(c)),
            iterator(std::end(c)),
            [](int, int) -> bool { throw std::runtime_error("test"); });
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

    caught_exception = false;
    try
    {
        hpx::ranges::is_sorted(policy, iterator(std::begin(c)),
            iterator(std::end(c)), std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
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
void test_sorted_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen() + 1);

    bool caught_exception = false;
    try
    {
        hpx::future<bool> f = hpx::ranges::is_sorted(p,
            decorated_iterator(
                std::begin(c), []() { throw std::runtime_error("test"); }),
            decorated_iterator(
                std::end(c), []() { throw std::runtime_error("test"); }));
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

    caught_exception = false;
    try
    {
        hpx::future<bool> f = hpx::ranges::is_sorted(p, iterator(std::begin(c)),
            iterator(std::end(c)),
            [](int, int) -> bool { throw std::runtime_error("test"); });
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

    HPX_TEST(caught_exception);

    caught_exception = false;
    try
    {
        hpx::future<bool> f = hpx::ranges::is_sorted(p, iterator(std::begin(c)),
            iterator(std::end(c)), std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
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
}

template <typename IteratorTag>
void test_sorted_exception_seq(IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;
    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), 0);

    bool caught_exception = false;
    try
    {
        hpx::ranges::is_sorted(decorated_iterator(std::begin(c),
                                   []() { throw std::runtime_error("test"); }),
            decorated_iterator(
                std::end(c), []() { throw std::runtime_error("test"); }));
    }
    catch (hpx::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions<hpx::execution::sequenced_policy,
            IteratorTag>::call(hpx::execution::seq, e);
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);

    caught_exception = false;
    try
    {
        hpx::ranges::is_sorted(iterator(std::begin(c)), iterator(std::end(c)),
            [](int, int) -> bool { throw std::runtime_error("test"); });
    }
    catch (hpx::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions<hpx::execution::sequenced_policy,
            IteratorTag>::call(hpx::execution::seq, e);
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);

    caught_exception = false;
    try
    {
        hpx::ranges::is_sorted(iterator(std::begin(c)), iterator(std::end(c)),
            std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
    }
    catch (hpx::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions<hpx::execution::sequenced_policy,
            IteratorTag>::call(hpx::execution::seq, e);
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename ExPolicy>
void test_sorted_exception(ExPolicy policy)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), 0);

    bool caught_exception = false;
    try
    {
        hpx::ranges::is_sorted(policy, c,
            [](int, int) -> bool { throw std::runtime_error("test"); });
    }
    catch (hpx::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions_base<ExPolicy>::call(policy, e);
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);

    caught_exception = false;
    try
    {
        hpx::ranges::is_sorted(policy, c, std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
    }
    catch (hpx::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions_base<ExPolicy>::call(policy, e);
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename ExPolicy>
void test_sorted_exception_async(ExPolicy p)
{
    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen() + 1);

    bool caught_exception = false;
    try
    {
        hpx::future<bool> f = hpx::ranges::is_sorted(
            p, c, [](int, int) -> bool { throw std::runtime_error("test"); });
        f.get();

        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions_base<ExPolicy>::call(p, e);
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);

    caught_exception = false;
    try
    {
        hpx::future<bool> f = hpx::ranges::is_sorted(p, c, std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
        f.get();

        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions_base<ExPolicy>::call(p, e);
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

void test_sorted_exception_seq()
{
    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), 0);

    bool caught_exception = false;
    try
    {
        hpx::ranges::is_sorted(
            c, [](int, int) -> int { throw std::runtime_error("test"); });
    }
    catch (hpx::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions_base<hpx::execution::sequenced_policy>::call(
            hpx::execution::seq, e);
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);

    caught_exception = false;
    try
    {
        hpx::ranges::is_sorted(c, std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
    }
    catch (hpx::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions_base<hpx::execution::sequenced_policy>::call(
            hpx::execution::seq, e);
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_sorted_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;
    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), 0);

    bool caught_bad_alloc = false;
    try
    {
        hpx::ranges::is_sorted(policy,
            decorated_iterator(
                std::begin(c), []() { throw std::runtime_error("test"); }),
            decorated_iterator(
                std::end(c), []() { throw std::runtime_error("test"); }));
    }
    catch (hpx::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);

    caught_bad_alloc = false;
    try
    {
        hpx::ranges::is_sorted(policy, iterator(std::begin(c)),
            iterator(std::end(c)),
            [](int, int) -> bool { throw std::runtime_error("test"); });
    }
    catch (hpx::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);

    caught_bad_alloc = false;
    try
    {
        hpx::ranges::is_sorted(policy, iterator(std::begin(c)),
            iterator(std::end(c)), std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
    }
    catch (hpx::exception_list const& e)
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
void test_sorted_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen() + 1);

    bool caught_bad_alloc = false;
    try
    {
        hpx::future<bool> f = hpx::ranges::is_sorted(p,
            decorated_iterator(
                std::begin(c), []() { throw std::runtime_error("test"); }),
            decorated_iterator(
                std::end(c), []() { throw std::runtime_error("test"); }));
        f.get();

        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);

    caught_bad_alloc = false;
    try
    {
        hpx::future<bool> f = hpx::ranges::is_sorted(p, iterator(std::begin(c)),
            iterator(std::end(c)),
            [](int, int) -> bool { throw std::runtime_error("test"); });
        f.get();

        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);

    HPX_TEST(caught_bad_alloc);

    caught_bad_alloc = false;
    try
    {
        hpx::future<bool> f = hpx::ranges::is_sorted(p, iterator(std::begin(c)),
            iterator(std::end(c)), std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
        f.get();

        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
}

template <typename IteratorTag>
void test_sorted_bad_alloc_seq(IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;
    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), 0);

    bool caught_bad_alloc = false;
    try
    {
        hpx::ranges::is_sorted(decorated_iterator(std::begin(c),
                                   []() { throw std::runtime_error("test"); }),
            decorated_iterator(
                std::end(c), []() { throw std::runtime_error("test"); }));
    }
    catch (hpx::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);

    caught_bad_alloc = false;
    try
    {
        hpx::ranges::is_sorted(iterator(std::begin(c)), iterator(std::end(c)),
            [](int, int) -> bool { throw std::runtime_error("test"); });
    }
    catch (hpx::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);

    caught_bad_alloc = false;
    try
    {
        hpx::ranges::is_sorted(iterator(std::begin(c)), iterator(std::end(c)),
            std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
    }
    catch (hpx::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
}

template <typename ExPolicy>
void test_sorted_bad_alloc(ExPolicy policy)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), 0);

    bool caught_bad_alloc = false;
    try
    {
        hpx::ranges::is_sorted(policy, c,
            [](int, int) -> bool { throw std::runtime_error("test"); });
    }
    catch (hpx::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);

    caught_bad_alloc = false;
    try
    {
        hpx::ranges::is_sorted(policy, c, std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
    }
    catch (hpx::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
}

template <typename ExPolicy>
void test_sorted_bad_alloc_async(ExPolicy p)
{
    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen() + 1);

    bool caught_bad_alloc = false;
    try
    {
        hpx::future<bool> f = hpx::ranges::is_sorted(
            p, c, [](int, int) -> bool { throw std::runtime_error("test"); });
        f.get();

        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);

    caught_bad_alloc = false;
    try
    {
        hpx::future<bool> f = hpx::ranges::is_sorted(p, c, std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
        f.get();

        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
}

void test_sorted_bad_alloc_seq()
{
    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), 0);

    bool caught_bad_alloc = false;
    try
    {
        hpx::ranges::is_sorted(
            c, [](int, int) -> int { throw std::runtime_error("test"); });
    }
    catch (hpx::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);

    caught_bad_alloc = false;
    try
    {
        hpx::ranges::is_sorted(c, std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
    }
    catch (hpx::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
}

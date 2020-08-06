//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/include/parallel_transform.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

struct add
{
    template <typename T1, typename T2>
    auto operator()(T1 const& v1, T2 const& v2) const -> decltype(v1 + v2)
    {
        return v1 + v2;
    }
};

struct throw_always
{
    template <typename T1, typename T2>
    auto operator()(T1 const& v1, T2 const& v2) const -> decltype(v1 + v2)
    {
        throw std::runtime_error("test");
    }
};

struct throw_bad_alloc
{
    template <typename T1, typename T2>
    auto operator()(T1 const& v1, T2 const& v2) const -> decltype(v1 + v2)
    {
        throw std::bad_alloc();
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_binary2(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c1(10007);
    std::vector<int> c2(c1.size());
    std::vector<int> d1(c1.size());    //-V656
    std::iota(std::begin(c1), std::end(c1),
        std::rand() % ((std::numeric_limits<int>::max)() / 2));
    std::iota(std::begin(c2), std::end(c2),
        std::rand() % ((std::numeric_limits<int>::max)() / 2));

    auto result = hpx::parallel::transform(policy, iterator(std::begin(c1)),
        iterator(std::end(c1)), std::begin(c2), std::end(c2), std::begin(d1),
        add());

    HPX_TEST(result.in1 == iterator(std::end(c1)));
    HPX_TEST(result.in2 == std::end(c2));
    HPX_TEST(result.out == std::end(d1));

    // verify values
    std::vector<int> d2(c1.size());
    std::transform(
        std::begin(c1), std::end(c1), std::begin(c2), std::begin(d2), add());

    std::size_t count = 0;
    HPX_TEST(std::equal(std::begin(d1), std::end(d1), std::begin(d2),
        [&count](int v1, int v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d2.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_transform_binary2_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c1(10007);
    std::vector<int> c2(c1.size());
    std::vector<int> d1(c1.size());    //-V656
    std::iota(std::begin(c1), std::end(c1),
        std::rand() % ((std::numeric_limits<int>::max)() / 2));
    std::iota(std::begin(c2), std::end(c2),
        std::rand() % ((std::numeric_limits<int>::max)() / 2));

    auto f = hpx::parallel::transform(p, iterator(std::begin(c1)),
        iterator(std::end(c1)), std::begin(c2), std::end(c2), std::begin(d1),
        add());
    f.wait();

    auto result = f.get();
    HPX_TEST(result.in1 == iterator(std::end(c1)));
    HPX_TEST(result.in2 == std::end(c2));
    HPX_TEST(result.out == std::end(d1));

    // verify values
    std::vector<int> d2(c1.size());
    std::transform(
        std::begin(c1), std::end(c1), std::begin(c2), std::begin(d2), add());

    std::size_t count = 0;
    HPX_TEST(std::equal(std::begin(d1), std::end(d1), std::begin(d2),
        [&count](int v1, int v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d2.size());
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_binary2_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c1(10007);
    std::vector<int> c2(c1.size());
    std::vector<int> d1(c1.size());    //-V656
    std::iota(std::begin(c1), std::end(c1), std::rand());
    std::iota(std::begin(c2), std::end(c2), std::rand());

    bool caught_exception = false;
    try
    {
        hpx::parallel::transform(policy, iterator(std::begin(c1)),
            iterator(std::end(c1)), std::begin(c2), std::end(c2),
            std::begin(d1), throw_always());

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
void test_transform_binary2_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c1(10007);
    std::vector<int> c2(c1.size());
    std::vector<int> d1(c1.size());    //-V656
    std::iota(std::begin(c1), std::end(c1), std::rand());
    std::iota(std::begin(c2), std::end(c2), std::rand());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = hpx::parallel::transform(p, iterator(std::begin(c1)),
            iterator(std::end(c1)), std::begin(c2), std::end(c2),
            std::begin(d1), throw_always());
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

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_binary2_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c1(10007);
    std::vector<int> c2(c1.size());
    std::vector<int> d1(c1.size());    //-V656
    std::iota(std::begin(c1), std::end(c1), std::rand());
    std::iota(std::begin(c2), std::end(c2), std::rand());

    bool caught_bad_alloc = false;
    try
    {
        hpx::parallel::transform(policy, iterator(std::begin(c1)),
            iterator(std::end(c1)), std::begin(c2), std::end(c2),
            std::begin(d1), throw_bad_alloc());

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
}

template <typename ExPolicy, typename IteratorTag>
void test_transform_binary2_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c1(10007);
    std::vector<int> c2(c1.size());
    std::vector<int> d1(c1.size());    //-V656
    std::iota(std::begin(c1), std::end(c1), std::rand());
    std::iota(std::begin(c2), std::end(c2), std::rand());

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = hpx::parallel::transform(p, iterator(std::begin(c1)),
            iterator(std::end(c1)), std::begin(c2), std::end(c2),
            std::begin(d1), throw_bad_alloc());
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

//  Copyright (c) 2017-2018 Taeguk Kwon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/include/parallel_unique.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/type_support/unused.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////

struct throw_always
{
    template <typename T>
    bool operator()(T const&, T const&) const
    {
        throw std::runtime_error("test");
    }
};

struct throw_bad_alloc
{
    template <typename T>
    bool operator()(T const&, T const&) const
    {
        throw std::bad_alloc();
    }
};

struct user_defined_type
{
    user_defined_type() = default;
    user_defined_type(int rand_no)
      : val(rand_no)
      , name(name_list[std::rand() % name_list.size()])
    {
    }

    bool operator<(user_defined_type const& t) const
    {
        if (this->name < t.name)
            return true;
        else if (this->name > t.name)
            return false;
        else
            return this->val < t.val;
    }

    bool operator>(user_defined_type const& t) const
    {
        if (this->name > t.name)
            return true;
        else if (this->name < t.name)
            return false;
        else
            return this->val > t.val;
    }

    bool operator==(user_defined_type const& t) const
    {
        return this->name == t.name && this->val == t.val;
    }

    bool operator!=(user_defined_type const& t) const
    {
        return this->name != t.name || this->val != t.val;
    }

    bool operator==(int rand_no) const
    {
        return this->val == rand_no;
    }

    bool operator!=(int rand_no) const
    {
        return this->val != rand_no;
    }

    static const std::vector<std::string> name_list;

    int val;
    std::string name;
};

const std::vector<std::string> user_defined_type::name_list{
    "ABB", "ABC", "ACB", "BASE", "CAA", "CAAA", "CAAB"};

struct random_fill
{
    random_fill() = default;
    random_fill(int rand_base, int range)
      : gen(std::rand())
      , dist(rand_base - range / 2, rand_base + range / 2)
    {
    }

    int operator()()
    {
        return dist(gen);
    }

    std::mt19937 gen;
    std::uniform_int_distribution<> dist;
};

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag, typename DataType,
    typename Pred>
void test_unique(
    ExPolicy policy, IteratorTag, DataType, Pred pred, int rand_base)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef typename std::vector<DataType>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t const size = 10007;
    std::vector<DataType> c(size), d;
    std::generate(std::begin(c), std::end(c), random_fill(rand_base, 6));
    d = c;

    auto result = hpx::parallel::unique(
        policy, iterator(std::begin(c)), iterator(std::end(c)), pred);
    auto solution = std::unique(std::begin(d), std::end(d), pred);

    bool equality =
        test::equal(std::begin(c), result.base(), std::begin(d), solution);

    HPX_TEST(equality);
}

template <typename ExPolicy, typename IteratorTag, typename DataType,
    typename Pred>
void test_unique_async(
    ExPolicy policy, IteratorTag, DataType, Pred pred, int rand_base)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef typename std::vector<DataType>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t const size = 10007;
    std::vector<DataType> c(size), d;
    std::generate(std::begin(c), std::end(c), random_fill(rand_base, 6));
    d = c;

    auto f = hpx::parallel::unique(
        policy, iterator(std::begin(c)), iterator(std::end(c)), pred);
    auto result = f.get();
    auto solution = std::unique(std::begin(d), std::end(d), pred);

    bool equality =
        test::equal(std::begin(c), result.base(), std::begin(d), solution);

    HPX_TEST(equality);
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_unique_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t const size = 10007;
    std::vector<int> c(size);
    std::generate(std::begin(c), std::end(c), random_fill());

    bool caught_exception = false;
    try
    {
        auto result = hpx::parallel::unique(policy, iterator(std::begin(c)),
            iterator(std::end(c)), throw_always());

        HPX_UNUSED(result);
        HPX_TEST(false);
    }
    catch (hpx::exception_list const&)
    {
        caught_exception = true;
        //test::test_num_exceptions<ExPolicy, IteratorTag>::call(policy, e);
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_unique_exception_async(ExPolicy policy, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t const size = 10007;
    std::vector<int> c(size);
    std::generate(std::begin(c), std::end(c), random_fill());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = hpx::parallel::unique(policy, iterator(std::begin(c)),
            iterator(std::end(c)), throw_always());
        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch (hpx::exception_list const&)
    {
        caught_exception = true;
        //test::test_num_exceptions<ExPolicy, IteratorTag>::call(policy, e);
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
void test_unique_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t const size = 10007;
    std::vector<int> c(size);
    std::generate(std::begin(c), std::end(c), random_fill());

    bool caught_bad_alloc = false;
    try
    {
        auto result = hpx::parallel::unique(policy, iterator(std::begin(c)),
            iterator(std::end(c)), throw_bad_alloc());

        HPX_UNUSED(result);
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
void test_unique_bad_alloc_async(ExPolicy policy, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t const size = 10007;
    std::vector<int> c(size);
    std::generate(std::begin(c), std::end(c), random_fill());

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = hpx::parallel::unique(policy, iterator(std::begin(c)),
            iterator(std::end(c)), throw_bad_alloc());
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

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag, typename DataType>
void test_unique_etc(ExPolicy policy, IteratorTag, DataType, int rand_base)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef typename std::vector<DataType>::iterator base_iterator;

    std::size_t const size = 10007;
    std::vector<DataType> c(size), d, org;
    std::generate(
        std::begin(c), std::end(c), random_fill(rand_base, size / 10));
    org = d = c;

    // Test default predicate.
    {
        typedef test::test_iterator<base_iterator, IteratorTag> iterator;

        auto result = hpx::parallel::unique(
            policy, iterator(std::begin(c)), iterator(std::end(c)));
        auto solution = std::unique(std::begin(d), std::end(d));

        bool equality =
            test::equal(std::begin(c), result.base(), std::begin(d), solution);

        HPX_TEST(equality);
    }

    // Test projection.
    {
        typedef test::test_iterator<base_iterator, IteratorTag> iterator;

        c = org;

        DataType val;
        auto result = hpx::parallel::unique(
            policy, iterator(std::begin(c)), iterator(std::end(c)),
            [](DataType const& a, DataType const& b) -> bool { return a == b; },
            [&val](DataType const&) -> DataType& {
                // This is projection.
                return val;
            });

        auto dist = std::distance(std::begin(c), result.base());
        HPX_TEST_EQ(dist, 1);
    }

    // Test sequential_unique with input_iterator_tag.
    {
        typedef test::test_iterator<base_iterator, std::input_iterator_tag>
            iterator;

        c = d = org;

        auto result = hpx::parallel::v1::detail::sequential_unique(
            iterator(std::begin(c)), iterator(std::end(c)),
            [](DataType const& a, DataType const& b) -> bool { return a == b; },
            [](DataType& t) -> DataType& { return t; });
        auto solution = std::unique(std::begin(d), std::end(d));

        bool equality =
            test::equal(std::begin(c), result.base(), std::begin(d), solution);

        HPX_TEST(equality);
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_unique()
{
    using namespace hpx::parallel;

    int rand_base = std::rand();

    ////////// Test cases for 'int' type.
    test_unique(
        execution::seq, IteratorTag(), int(),
        [](const int a, const int b) -> bool { return a == b; }, rand_base);
    test_unique(
        execution::par, IteratorTag(), int(),
        [rand_base](const int a, const int b) -> bool {
            return a == b && b == rand_base;
        },
        rand_base);
    test_unique(
        execution::par_unseq, IteratorTag(), int(),
        [](const int a, const int b) -> bool { return a == b; }, rand_base);

    ////////// Test cases for user defined type.
    test_unique(
        execution::seq, IteratorTag(), user_defined_type(),
        [](user_defined_type const& a, user_defined_type const& b) -> bool {
            return a == b;
        },
        rand_base);
    test_unique(
        execution::par, IteratorTag(), user_defined_type(),
        [](user_defined_type const& a, user_defined_type const& b) -> bool {
            return a == b;
        },
        rand_base);
    test_unique(
        execution::par_unseq, IteratorTag(), user_defined_type(),
        [rand_base](user_defined_type const& a, user_defined_type const& b)
            -> bool { return a == b && b == rand_base; },
        rand_base);

    ////////// Asynchronous test cases for 'int' type.
    test_unique_async(
        execution::seq(execution::task), IteratorTag(), int(),
        [rand_base](const int a, const int b) -> bool {
            return a == b && b == rand_base;
        },
        rand_base);
    test_unique_async(
        execution::par(execution::task), IteratorTag(), int(),
        [](const int a, const int b) -> bool { return a == b; }, rand_base);

    ////////// Asynchronous test cases for user defined type.
    test_unique_async(
        execution::seq(execution::task), IteratorTag(), user_defined_type(),
        [](user_defined_type const& a, user_defined_type const& b) -> bool {
            return a == b;
        },
        rand_base);
    test_unique_async(
        execution::par(execution::task), IteratorTag(), user_defined_type(),
        [rand_base](user_defined_type const& a, user_defined_type const& b)
            -> bool { return a == rand_base && b == rand_base; },
        rand_base);

    ////////// Corner test cases.
    test_unique(
        execution::par, IteratorTag(), int(),
        [](const int, const int) -> bool { return true; }, rand_base);
    test_unique(
        execution::par_unseq, IteratorTag(), user_defined_type(),
        [](user_defined_type const&, user_defined_type const&) -> bool {
            return false;
        },
        rand_base);

    ////////// Another test cases for justifying the implementation.
    test_unique_etc(
        execution::seq, IteratorTag(), user_defined_type(), rand_base);
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_unique_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_unique_exception(execution::seq, IteratorTag());
    test_unique_exception(execution::par, IteratorTag());

    test_unique_exception_async(execution::seq(execution::task), IteratorTag());
    test_unique_exception_async(execution::par(execution::task), IteratorTag());
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_unique_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_unique_bad_alloc(execution::seq, IteratorTag());
    test_unique_bad_alloc(execution::par, IteratorTag());

    test_unique_bad_alloc_async(execution::seq(execution::task), IteratorTag());
    test_unique_bad_alloc_async(execution::par(execution::task), IteratorTag());
}

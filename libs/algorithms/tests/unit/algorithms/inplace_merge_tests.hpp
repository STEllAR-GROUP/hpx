//  Copyright (c) 2017-2018 Taeguk Kwon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/include/parallel_merge.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/type_support/unused.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
int seed = std::random_device{}();
std::mt19937 _gen(seed);

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
    {
        std::uniform_int_distribution<> dist(0, name_list.size() - 1);
        name = name_list[dist(_gen)];
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

    user_defined_type operator+(int val) const
    {
        user_defined_type t(*this);
        t.val += val;
        return t;
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
      : gen(_gen())
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
    typename Comp>
void test_inplace_merge(
    ExPolicy policy, IteratorTag, DataType, Comp comp, int rand_base)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef typename std::vector<DataType>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    using hpx::util::get;

    std::size_t const left_size = 300007, right_size = 123456;
    std::vector<DataType> res(left_size + right_size), sol;

    base_iterator res_first = std::begin(res);
    base_iterator res_middle = res_first + left_size;
    base_iterator res_last = std::end(res);

    std::generate(res_first, res_middle, random_fill(rand_base, 6));
    std::generate(res_middle, res_last, random_fill(rand_base, 8));
    std::sort(res_first, res_middle, comp);
    std::sort(res_middle, res_last, comp);

    sol = res;
    base_iterator sol_first = std::begin(sol);
    base_iterator sol_middle = sol_first + left_size;
    base_iterator sol_last = std::end(sol);

    hpx::parallel::inplace_merge(policy, iterator(res_first),
        iterator(res_middle), iterator(res_last), comp);
    std::inplace_merge(sol_first, sol_middle, sol_last, comp);

    bool equality = test::equal(res_first, res_last, sol_first, sol_last);

    HPX_TEST(equality);
}

template <typename ExPolicy, typename IteratorTag, typename DataType,
    typename Comp>
void test_inplace_merge_async(
    ExPolicy policy, IteratorTag, DataType, Comp comp, int rand_base)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef typename std::vector<DataType>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    using hpx::util::get;

    std::size_t const left_size = 300007, right_size = 123456;
    std::vector<DataType> res(left_size + right_size), sol;

    base_iterator res_first = std::begin(res);
    base_iterator res_middle = res_first + left_size;
    base_iterator res_last = std::end(res);

    std::generate(res_first, res_middle, random_fill(rand_base, 6));
    std::generate(res_middle, res_last, random_fill(rand_base, 8));
    std::sort(res_first, res_middle, comp);
    std::sort(res_middle, res_last, comp);

    sol = res;
    base_iterator sol_first = std::begin(sol);
    base_iterator sol_middle = sol_first + left_size;
    base_iterator sol_last = std::end(sol);

    auto f = hpx::parallel::inplace_merge(policy, iterator(res_first),
        iterator(res_middle), iterator(res_last), comp);
    f.get();
    std::inplace_merge(sol_first, sol_middle, sol_last, comp);

    bool equality = test::equal(res_first, res_last, sol_first, sol_last);

    HPX_TEST(equality);
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_inplace_merge_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t const left_size = 300007, right_size = 123456;
    std::vector<int> res(left_size + right_size);

    base_iterator res_first = std::begin(res);
    base_iterator res_middle = res_first + left_size;
    base_iterator res_last = std::end(res);

    std::generate(res_first, res_middle, random_fill());
    std::generate(res_middle, res_last, random_fill());
    std::sort(res_first, res_middle);
    std::sort(res_middle, res_last);

    bool caught_exception = false;
    try
    {
        auto result = hpx::parallel::inplace_merge(policy, iterator(res_first),
            iterator(res_middle), iterator(res_last), throw_always());

        HPX_UNUSED(result);
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
void test_inplace_merge_exception_async(ExPolicy policy, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t const left_size = 300007, right_size = 123456;
    std::vector<int> res(left_size + right_size);

    base_iterator res_first = std::begin(res);
    base_iterator res_middle = res_first + left_size;
    base_iterator res_last = std::end(res);

    std::generate(res_first, res_middle, random_fill());
    std::generate(res_middle, res_last, random_fill());
    std::sort(res_first, res_middle);
    std::sort(res_middle, res_last);

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = hpx::parallel::inplace_merge(policy, iterator(res_first),
            iterator(res_middle), iterator(res_last), throw_always());
        returned_from_algorithm = true;
        f.get();

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
    HPX_TEST(returned_from_algorithm);
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_inplace_merge_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t const left_size = 300007, right_size = 123456;
    std::vector<int> res(left_size + right_size);

    base_iterator res_first = std::begin(res);
    base_iterator res_middle = res_first + left_size;
    base_iterator res_last = std::end(res);

    std::generate(res_first, res_middle, random_fill());
    std::generate(res_middle, res_last, random_fill());
    std::sort(res_first, res_middle);
    std::sort(res_middle, res_last);

    bool caught_bad_alloc = false;
    try
    {
        auto result = hpx::parallel::inplace_merge(policy, iterator(res_first),
            iterator(res_middle), iterator(res_last), throw_bad_alloc());

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
void test_inplace_merge_bad_alloc_async(ExPolicy policy, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t const left_size = 300007, right_size = 123456;
    std::vector<int> res(left_size + right_size);

    base_iterator res_first = std::begin(res);
    base_iterator res_middle = res_first + left_size;
    base_iterator res_last = std::end(res);

    std::generate(res_first, res_middle, random_fill());
    std::generate(res_middle, res_last, random_fill());
    std::sort(res_first, res_middle);
    std::sort(res_middle, res_last);

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = hpx::parallel::inplace_merge(policy, iterator(res_first),
            iterator(res_middle), iterator(res_last), throw_bad_alloc());
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
void test_inplace_merge_etc(
    ExPolicy policy, IteratorTag, DataType, int rand_base)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef typename std::vector<DataType>::iterator base_iterator;

    using hpx::util::get;

    std::size_t const left_size = 300007, right_size = 123456;
    std::vector<DataType> res(left_size + right_size), sol, org;

    base_iterator res_first = std::begin(res);
    base_iterator res_middle = res_first + left_size;
    base_iterator res_last = std::end(res);

    std::generate(res_first, res_middle, random_fill(rand_base, 6));
    std::generate(res_middle, res_last, random_fill(rand_base, 8));
    std::sort(res_first, res_middle);
    std::sort(res_middle, res_last);

    org = sol = res;
    base_iterator sol_first = std::begin(sol);
    base_iterator sol_middle = sol_first + left_size;
    base_iterator sol_last = std::end(sol);

    // Test default comparison.
    {
        typedef test::test_iterator<base_iterator, IteratorTag> iterator;

        sol = res = org;

        hpx::parallel::inplace_merge(policy, iterator(res_first),
            iterator(res_middle), iterator(res_last));
        std::inplace_merge(sol_first, sol_middle, sol_last);

        bool equality = test::equal(res_first, res_last, sol_first, sol_last);

        HPX_TEST(equality);
    }

    // Test projection.
    {
#if defined(HPX_HAVE_STABLE_INPLACE_MERGE)
        typedef test::test_iterator<base_iterator, IteratorTag> iterator;

        sol = res = org;

        DataType val;
        hpx::parallel::inplace_merge(
            policy, iterator(res_first), iterator(res_middle),
            iterator(res_last),
            [](DataType const& a, DataType const& b) -> bool { return a < b; },
            [&val](DataType const& elem) -> DataType& {
                // This is projection.
                return val;
            });

        // The container must not be changed.
        bool equality = test::equal(res_first, res_last, sol_first, sol_last);

        HPX_TEST(equality);
#endif
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag, typename DataType>
void test_inplace_merge_stable(
    ExPolicy policy, IteratorTag, DataType, int rand_base)
{
#if defined(HPX_HAVE_STABLE_INPLACE_MERGE)
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef typename std::pair<DataType, int> ElemType;
    typedef typename std::vector<ElemType>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    using hpx::util::get;

    std::size_t const left_size = 300007, right_size = 123456;
    std::vector<ElemType> res(left_size + right_size);

    base_iterator res_first = std::begin(res);
    base_iterator res_middle = res_first + left_size;
    base_iterator res_last = std::end(res);

    int no = 0;
    auto rf = random_fill(rand_base, 6);
    std::generate(res_first, res_middle, [&no, &rf]() -> std::pair<int, int> {
        return {rf(), no++};
    });
    rf = random_fill(rand_base, 8);
    std::generate(res_middle, res_last, [&no, &rf]() -> std::pair<int, int> {
        return {rf(), no++};
    });
    std::sort(res_first, res_middle);
    std::sort(res_middle, res_last);

    hpx::parallel::inplace_merge(
        policy, iterator(res_first), iterator(res_middle), iterator(res_last),
        [](DataType const& a, DataType const& b) -> bool { return a < b; },
        [](ElemType const& elem) -> DataType const& {
            // This is projection.
            return elem.first;
        });

    bool stable = true;
    int check_count = 0;
    for (std::size_t i = 1u; i < left_size + right_size; ++i)
    {
        if (res[i - 1].first == res[i].first)
        {
            ++check_count;
            if (res[i - 1].second > res[i].second)
                stable = false;
        }
    }

    bool test_is_meaningful = check_count >= 100;

    HPX_TEST(test_is_meaningful);
    HPX_TEST(stable);
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_inplace_merge()
{
    using namespace hpx::parallel;

    int rand_base = _gen();

    ////////// Test cases for 'int' type.
    test_inplace_merge(
        execution::seq, IteratorTag(), int(),
        [](const int a, const int b) -> bool { return a < b; }, rand_base);
    test_inplace_merge(
        execution::par, IteratorTag(), int(),
        [](const int a, const int b) -> bool { return a < b; }, rand_base);
    test_inplace_merge(
        execution::par_unseq, IteratorTag(), int(),
        [](const int a, const int b) -> bool { return a > b; }, rand_base);

    ////////// Test cases for user defined type.
    test_inplace_merge(
        execution::seq, IteratorTag(), user_defined_type(),
        [](user_defined_type const& a, user_defined_type const& b) -> bool {
            return a < b;
        },
        rand_base);
    test_inplace_merge(
        execution::par, IteratorTag(), user_defined_type(),
        [](user_defined_type const& a, user_defined_type const& b) -> bool {
            return a > b;
        },
        rand_base);
    test_inplace_merge(
        execution::par_unseq, IteratorTag(), user_defined_type(),
        [](user_defined_type const& a, user_defined_type const& b) -> bool {
            return a < b;
        },
        rand_base);

    ////////// Asynchronous test cases for 'int' type.
    test_inplace_merge_async(
        execution::seq(execution::task), IteratorTag(), int(),
        [](const int a, const int b) -> bool { return a > b; }, rand_base);
    test_inplace_merge_async(
        execution::par(execution::task), IteratorTag(), int(),
        [](const int a, const int b) -> bool { return a > b; }, rand_base);

    ////////// Asynchronous test cases for user defined type.
    test_inplace_merge_async(
        execution::seq(execution::task), IteratorTag(), user_defined_type(),
        [](user_defined_type const& a, user_defined_type const& b) -> bool {
            return a < b;
        },
        rand_base);
    test_inplace_merge_async(
        execution::par(execution::task), IteratorTag(), user_defined_type(),
        [](user_defined_type const& a, user_defined_type const& b) -> bool {
            return a < b;
        },
        rand_base);

    ////////// Another test cases for justifying the implementation.
    test_inplace_merge_etc(
        execution::seq, IteratorTag(), user_defined_type(), rand_base);
    test_inplace_merge_etc(
        execution::par, IteratorTag(), user_defined_type(), rand_base);
    test_inplace_merge_etc(
        execution::par_unseq, IteratorTag(), user_defined_type(), rand_base);

    ////////// Test cases for checking whether the algorithm is stable.
    test_inplace_merge_stable(execution::seq, IteratorTag(), int(), rand_base);
    test_inplace_merge_stable(execution::par, IteratorTag(), int(), rand_base);
    test_inplace_merge_stable(
        execution::par_unseq, IteratorTag(), int(), rand_base);
    test_inplace_merge_stable(
        execution::seq, IteratorTag(), user_defined_type(), rand_base);
    test_inplace_merge_stable(
        execution::par, IteratorTag(), user_defined_type(), rand_base);
    test_inplace_merge_stable(
        execution::par_unseq, IteratorTag(), user_defined_type(), rand_base);
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_inplace_merge_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_inplace_merge_exception(execution::seq, IteratorTag());
    test_inplace_merge_exception(execution::par, IteratorTag());

    test_inplace_merge_exception_async(
        execution::seq(execution::task), IteratorTag());
    test_inplace_merge_exception_async(
        execution::par(execution::task), IteratorTag());
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_inplace_merge_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_inplace_merge_bad_alloc(execution::seq, IteratorTag());
    test_inplace_merge_bad_alloc(execution::par, IteratorTag());

    test_inplace_merge_bad_alloc_async(
        execution::seq(execution::task), IteratorTag());
    test_inplace_merge_bad_alloc_async(
        execution::par(execution::task), IteratorTag());
}

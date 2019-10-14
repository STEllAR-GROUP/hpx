//  Copyright (c) 2017-2018 Taeguk Kwon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TEST_REMOVE_DEC_24_2017_0820PM)
#define HPX_PARALLEL_TEST_REMOVE_DEC_24_2017_0820PM

#include <hpx/include/parallel_remove.hpp>
#include <hpx/testing.hpp>
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
unsigned int seed = std::random_device{}();
std::mt19937 g(seed);

struct throw_always
{
    template <typename T>
    bool operator()(T const&) const
    {
        throw std::runtime_error("test");
    }
};

struct throw_bad_alloc
{
    template <typename T>
    bool operator()(T const&) const
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
        std::uniform_int_distribution<> dis(0, name_list.size() - 1);
        name = name_list[dis(g)];
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

    bool operator==(int rand_no) const
    {
        return this->val == rand_no;
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
      : gen(g())
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
    typename ValueType>
void test_remove(
    ExPolicy policy, IteratorTag, DataType, ValueType value, int rand_base)
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

    auto result = hpx::parallel::remove(
        policy, iterator(std::begin(c)), iterator(std::end(c)), value);
    auto solution = std::remove(std::begin(d), std::end(d), value);

    bool equality =
        test::equal(std::begin(c), result.base(), std::begin(d), solution);

    HPX_TEST(equality);
}

template <typename ExPolicy, typename IteratorTag, typename DataType,
    typename ValueType>
void test_remove_async(
    ExPolicy policy, IteratorTag, DataType, ValueType value, int rand_base)
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

    auto f = hpx::parallel::remove(
        policy, iterator(std::begin(c)), iterator(std::end(c)), value);
    auto result = f.get();
    auto solution = std::remove(std::begin(d), std::end(d), value);

    bool equality =
        test::equal(std::begin(c), result.base(), std::begin(d), solution);

    HPX_TEST(equality);
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag, typename DataType,
    typename Pred>
void test_remove_if(
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

    auto result = hpx::parallel::remove_if(
        policy, iterator(std::begin(c)), iterator(std::end(c)), pred);
    auto solution = std::remove_if(std::begin(d), std::end(d), pred);

    bool equality =
        test::equal(std::begin(c), result.base(), std::begin(d), solution);

    HPX_TEST(equality);
}

template <typename ExPolicy, typename IteratorTag, typename DataType,
    typename Pred>
void test_remove_if_async(
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

    auto f = hpx::parallel::remove_if(
        policy, iterator(std::begin(c)), iterator(std::end(c)), pred);
    auto result = f.get();
    auto solution = std::remove_if(std::begin(d), std::end(d), pred);

    bool equality =
        test::equal(std::begin(c), result.base(), std::begin(d), solution);

    HPX_TEST(equality);
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_remove_exception(
    ExPolicy policy, IteratorTag, bool test_for_remove_if = false)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::size_t const size = 10007;
    std::vector<int> c(size);
    std::generate(std::begin(c), std::end(c), random_fill());

    bool caught_exception = false;
    try
    {
        if (test_for_remove_if)
        {
            hpx::parallel::remove_if(policy, decorated_iterator(std::begin(c)),
                decorated_iterator(std::end(c)), throw_always());
        }
        else
        {
            hpx::parallel::remove(policy,
                decorated_iterator(
                    std::begin(c), []() { throw std::runtime_error("test"); }),
                decorated_iterator(std::end(c)), int(10));
        }

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
void test_remove_exception_async(
    ExPolicy policy, IteratorTag, bool test_for_remove_if = false)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::size_t const size = 10007;
    std::vector<int> c(size);
    std::generate(std::begin(c), std::end(c), random_fill());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        hpx::future<decorated_iterator> f;

        if (test_for_remove_if)
        {
            f = hpx::parallel::remove_if(policy,
                decorated_iterator(std::begin(c)),
                decorated_iterator(std::end(c)), throw_always());
        }
        else
        {
            f = hpx::parallel::remove(policy,
                decorated_iterator(
                    std::begin(c), []() { throw std::runtime_error("test"); }),
                decorated_iterator(std::end(c)), int(10));
        }

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
void test_remove_bad_alloc(
    ExPolicy policy, IteratorTag, bool test_for_remove_if = false)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::size_t const size = 10007;
    std::vector<int> c(size);
    std::generate(std::begin(c), std::end(c), random_fill());

    bool caught_bad_alloc = false;
    try
    {
        if (test_for_remove_if)
        {
            hpx::parallel::remove_if(policy, decorated_iterator(std::begin(c)),
                decorated_iterator(std::end(c)), throw_bad_alloc());
        }
        else
        {
            hpx::parallel::remove(policy,
                decorated_iterator(
                    std::begin(c), []() { throw std::bad_alloc(); }),
                decorated_iterator(std::end(c)), int(10));
        }

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
void test_remove_bad_alloc_async(
    ExPolicy policy, IteratorTag, bool test_for_remove_if = false)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::size_t const size = 10007;
    std::vector<int> c(size);
    std::generate(std::begin(c), std::end(c), random_fill());

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try
    {
        hpx::future<decorated_iterator> f;

        if (test_for_remove_if)
        {
            f = hpx::parallel::remove_if(policy,
                decorated_iterator(std::begin(c)),
                decorated_iterator(std::end(c)), throw_bad_alloc());
        }
        else
        {
            f = hpx::parallel::remove(policy,
                decorated_iterator(
                    std::begin(c), []() { throw std::bad_alloc(); }),
                decorated_iterator(std::end(c)), int(10));
        }

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
void test_remove_etc(ExPolicy policy, IteratorTag, DataType, int rand_base,
    bool test_for_remove_if = false)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef typename std::vector<DataType>::iterator base_iterator;

    std::size_t const size = 10007;
    std::vector<DataType> c(size), org;
    std::generate(
        std::begin(c), std::end(c), random_fill(rand_base, size / 10));
    org = c;

    // Test projection.
    {
        typedef test::test_iterator<base_iterator, IteratorTag> iterator;

        c = org;

        DataType value(rand_base);
        iterator result;

        if (test_for_remove_if)
        {
            result = hpx::parallel::remove_if(
                policy, iterator(std::begin(c)), iterator(std::end(c)),
                [&value](DataType const& a) -> bool { return a == value; },
                [&value](DataType const&) -> DataType& {
                    // This is projection.
                    return value;
                });
        }
        else
        {
            result = hpx::parallel::remove(policy, iterator(std::begin(c)),
                iterator(std::end(c)), value,
                [&value](DataType const&) -> DataType& {
                    // This is projection.
                    return value;
                });
        }

        auto dist = std::distance(std::begin(c), result.base());
        HPX_TEST_EQ(dist, 0);
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_remove(IteratorTag, int rand_base)
{
    using namespace hpx::parallel;

    ////////// Test cases for 'int' type.
    test_remove(
        execution::seq, IteratorTag(), int(), int(rand_base + 1), rand_base);
    test_remove(
        execution::par, IteratorTag(), int(), int(rand_base), rand_base);
    test_remove(execution::par_unseq, IteratorTag(), int(), int(rand_base - 1),
        rand_base);

    ////////// Test cases for user defined type.
    test_remove(execution::seq, IteratorTag(), user_defined_type(),
        user_defined_type(rand_base), rand_base);
    test_remove(execution::par, IteratorTag(), user_defined_type(),
        user_defined_type(rand_base + 1), rand_base);

    ////////// Asynchronous test cases for 'int' type.
    test_remove_async(execution::seq(execution::task), IteratorTag(), int(),
        int(rand_base), rand_base);
    test_remove_async(execution::par(execution::task), IteratorTag(), int(),
        int(rand_base - 1), rand_base);

    ////////// Asynchronous test cases for user defined type.
    test_remove_async(execution::seq(execution::task), IteratorTag(),
        user_defined_type(), user_defined_type(rand_base - 1), rand_base);
    test_remove_async(execution::par(execution::task), IteratorTag(),
        user_defined_type(), user_defined_type(rand_base), rand_base);
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_remove_if(IteratorTag, int rand_base)
{
    using namespace hpx::parallel;

    ////////// Test cases for 'int' type.
    test_remove_if(
        execution::seq, IteratorTag(), int(),
        [rand_base](const int a) -> bool { return a == rand_base; }, rand_base);
    test_remove_if(
        execution::par, IteratorTag(), int(),
        [rand_base](const int a) -> bool { return !(a == rand_base); },
        rand_base);
    test_remove_if(
        execution::par_unseq, IteratorTag(), int(),
        [rand_base](const int a) -> bool { return a == rand_base; }, rand_base);

    ////////// Test cases for user defined type.
    test_remove_if(
        execution::seq, IteratorTag(), user_defined_type(),
        [rand_base](
            user_defined_type const& a) -> bool { return !(a == rand_base); },
        rand_base);
    test_remove_if(
        execution::par, IteratorTag(), user_defined_type(),
        [rand_base](
            user_defined_type const& a) -> bool { return a == rand_base; },
        rand_base);
    test_remove_if(
        execution::par_unseq, IteratorTag(), user_defined_type(),
        [rand_base](
            user_defined_type const& a) -> bool { return !(a == rand_base); },
        rand_base);

    ////////// Asynchronous test cases for 'int' type.
    test_remove_if_async(
        execution::seq(execution::task), IteratorTag(), int(),
        [rand_base](const int a) -> bool { return !(a == rand_base); },
        rand_base);
    test_remove_if_async(
        execution::par(execution::task), IteratorTag(), int(),
        [rand_base](const int a) -> bool { return a == rand_base; }, rand_base);

    ////////// Asynchronous test cases for user defined type.
    test_remove_if_async(
        execution::seq(execution::task), IteratorTag(), user_defined_type(),
        [rand_base](
            user_defined_type const& a) -> bool { return a == rand_base; },
        rand_base);
    test_remove_if_async(
        execution::par(execution::task), IteratorTag(), user_defined_type(),
        [rand_base](
            user_defined_type const& a) -> bool { return !(a == rand_base); },
        rand_base);

    ////////// Corner test cases.
    test_remove_if(
        execution::par, IteratorTag(), int(),
        [](const int) -> bool { return true; }, rand_base);
    test_remove_if(
        execution::par_unseq, IteratorTag(), user_defined_type(),
        [](user_defined_type const&) -> bool { return false; }, rand_base);
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_remove(bool test_for_remove_if = false)
{
    int rand_base = g();

    if (test_for_remove_if)
    {
        test_remove_if(IteratorTag(), rand_base);
    }
    else
    {
        test_remove(IteratorTag(), rand_base);
    }

    ////////// Another test cases for justifying the implementation.
    test_remove_etc(hpx::parallel::execution::seq, IteratorTag(),
        user_defined_type(), rand_base, test_for_remove_if);
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_remove_exception(bool test_for_remove_if = false)
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_remove_exception(execution::seq, IteratorTag(), test_for_remove_if);
    test_remove_exception(execution::par, IteratorTag(), test_for_remove_if);

    test_remove_exception_async(
        execution::seq(execution::task), IteratorTag(), test_for_remove_if);
    test_remove_exception_async(
        execution::par(execution::task), IteratorTag(), test_for_remove_if);
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_remove_bad_alloc(bool test_for_remove_if = false)
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_remove_bad_alloc(execution::seq, IteratorTag(), test_for_remove_if);
    test_remove_bad_alloc(execution::par, IteratorTag(), test_for_remove_if);

    test_remove_bad_alloc_async(
        execution::seq(execution::task), IteratorTag(), test_for_remove_if);
    test_remove_bad_alloc_async(
        execution::par(execution::task), IteratorTag(), test_for_remove_if);
}

#endif

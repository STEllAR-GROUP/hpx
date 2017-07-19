//  Copyright (c) 2017 Taeguk Kwon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TEST_PARTITION_COPY_JUN_18_2017_0245AM)
#define HPX_PARALLEL_TEST_PARTITION_COPY_JUN_18_2017_0245AM

#include <hpx/include/parallel_partition.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/unused.hpp>

#include <boost/random.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
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
        : val(rand_no),
        name(name_list[std::rand() % name_list.size()])
    {}

    bool operator<(int rand_base) const
    {
        static std::string const base_name = "BASE";

        if (this->name < base_name)
            return true;
        else if (this->name > base_name)
            return false;
        else
            return this->val < rand_base;
    }

    bool operator==(user_defined_type const& t) const
    {
        return this->name == t.name && this->val == t.val;
    }

    static const std::vector<std::string> name_list;

    int val;
    std::string name;
};

const std::vector<std::string> user_defined_type::name_list{
    "ABB", "ABC", "ACB", "BASE", "CAA", "CAAA", "CAAB"
};

struct random_fill
{
    random_fill(int rand_base, int range)
        : gen(std::rand()),
        dist(rand_base - range / 2, rand_base + range / 2)
    {}

    int operator()()
    {
        return dist(gen);
    }

    boost::random::mt19937 gen;
    boost::random::uniform_int_distribution<> dist;
};

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag, typename DataType, typename Pred>
void test_partition_copy(ExPolicy policy, IteratorTag, DataType, Pred pred,
    int rand_base)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef typename std::vector<DataType>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    using hpx::util::get;

    std::size_t const size = 10007;
    std::vector<DataType> c(size),
        d_true_res(size), d_false_res(size),
        d_true_sol(size), d_false_sol(size);
    std::generate(std::begin(c), std::end(c), random_fill(rand_base, size / 10));

    auto result = hpx::parallel::partition_copy(policy,
        iterator(std::begin(c)), iterator(std::end(c)),
        iterator(std::begin(d_true_res)), iterator(std::begin(d_false_res)),
        pred);
    auto solution = std::partition_copy(std::begin(c), std::end(c),
        std::begin(d_true_sol), std::begin(d_false_sol),
        pred);

    HPX_TEST(get<0>(result).base() == std::end(c));

    bool equality_true = std::equal(
        std::begin(d_true_res), get<1>(result).base(),
        std::begin(d_true_sol), get<0>(solution));
    bool equality_false = std::equal(
        std::begin(d_false_res), get<2>(result).base(),
        std::begin(d_false_sol), get<1>(solution));

    HPX_TEST(equality_true);
    HPX_TEST(equality_false);
}

template <typename ExPolicy, typename IteratorTag, typename DataType, typename Pred>
void test_partition_copy_async(ExPolicy policy, IteratorTag, DataType, Pred pred,
    int rand_base)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef typename std::vector<DataType>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    using hpx::util::get;

    std::size_t const size = 10007;
    std::vector<DataType> c(size),
        d_true_res(size), d_false_res(size),
        d_true_sol(size), d_false_sol(size);
    std::generate(std::begin(c), std::end(c), random_fill(rand_base, size / 10));

    auto f = hpx::parallel::partition_copy(policy,
        iterator(std::begin(c)), iterator(std::end(c)),
        iterator(std::begin(d_true_res)), iterator(std::begin(d_false_res)),
        pred);
    auto result = f.get();
    auto solution = std::partition_copy(std::begin(c), std::end(c),
        std::begin(d_true_sol), std::begin(d_false_sol),
        pred);

    HPX_TEST(get<0>(result).base() == std::end(c));

    bool equality_true = std::equal(
        std::begin(d_true_res), get<1>(result).base(),
        std::begin(d_true_sol), get<0>(solution));
    bool equality_false = std::equal(
        std::begin(d_false_res), get<2>(result).base(),
        std::begin(d_false_sol), get<1>(solution));

    HPX_TEST(equality_true);
    HPX_TEST(equality_false);
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_partition_copy_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t const size = 10007;
    std::vector<int> c(size), d_true(size), d_false(size);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    try {
        auto result = hpx::parallel::partition_copy(policy,
            iterator(std::begin(c)), iterator(std::end(c)),
            iterator(std::begin(d_true)), iterator(std::begin(d_false)),
            throw_always());

        HPX_UNUSED(result);
        HPX_TEST(false);
    }
    catch(hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(policy, e);
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_partition_copy_exception_async(ExPolicy policy, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t const size = 10007;
    std::vector<int> c(size), d_true(size), d_false(size);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        auto f = hpx::parallel::partition_copy(policy,
            iterator(std::begin(c)), iterator(std::end(c)),
            iterator(std::begin(d_true)), iterator(std::begin(d_false)),
            throw_always());
        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch(hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(policy, e);
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST(returned_from_algorithm);
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_partition_copy_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t const size = 10007;
    std::vector<int> c(size), d_true(size), d_false(size);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_bad_alloc = false;
    try {
        auto result = hpx::parallel::partition_copy(policy,
            iterator(std::begin(c)), iterator(std::end(c)),
            iterator(std::begin(d_true)), iterator(std::begin(d_false)),
            throw_bad_alloc());

        HPX_UNUSED(result);
        HPX_TEST(false);
    }
    catch(std::bad_alloc const&) {
        caught_bad_alloc = true;
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
}

template <typename ExPolicy, typename IteratorTag>
void test_partition_copy_bad_alloc_async(ExPolicy policy, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t const size = 10007;
    std::vector<int> c(size), d_true(size), d_false(size);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        auto f = hpx::parallel::partition_copy(policy,
            iterator(std::begin(c)), iterator(std::end(c)),
            iterator(std::begin(d_true)), iterator(std::begin(d_false)),
            throw_bad_alloc());
        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch(std::bad_alloc const&) {
        caught_bad_alloc = true;
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
    HPX_TEST(returned_from_algorithm);
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_partition_copy()
{
    using namespace hpx::parallel;

    int rand_base = std::rand();

    ////////// Test cases for 'int' type.
    test_partition_copy(execution::seq, IteratorTag(), int(),
        [rand_base](const int n) -> bool {
            return n < rand_base;
        }, rand_base);
    test_partition_copy(execution::par, IteratorTag(), int(),
        [rand_base](const int n) -> bool {
            return n <= rand_base;
        }, rand_base);
    test_partition_copy(execution::par_unseq, IteratorTag(), int(),
        [rand_base](const int n) -> bool {
            return n > rand_base;
        }, rand_base);

    ////////// Test cases for user defined type.
    test_partition_copy(execution::seq, IteratorTag(), user_defined_type(),
        [rand_base](user_defined_type const& t) -> bool {
            return t < rand_base;
        }, rand_base);
    test_partition_copy(execution::par, IteratorTag(), user_defined_type(),
        [rand_base](user_defined_type const& t) -> bool {
            return !(t < rand_base);
        }, rand_base);
    test_partition_copy(execution::par_unseq, IteratorTag(), user_defined_type(),
        [rand_base](user_defined_type const& t) -> bool {
            return t < rand_base;
        }, rand_base);

    ////////// Asynchronous test cases for 'int' type.
    test_partition_copy_async(execution::seq(execution::task), IteratorTag(), int(),
        [rand_base](const int n) -> bool {
            return n >= rand_base;
        }, rand_base);
    test_partition_copy_async(execution::par(execution::task), IteratorTag(), int(),
        [rand_base](const int n) -> bool {
            return n < rand_base;
        }, rand_base);

    ////////// Asynchronous test cases for user defined type.
    test_partition_copy_async(execution::seq(execution::task), IteratorTag(),
        user_defined_type(),
        [rand_base](user_defined_type const& t) -> bool {
            return !(t < rand_base);
        }, rand_base);
    test_partition_copy_async(execution::par(execution::task), IteratorTag(),
        user_defined_type(),
        [rand_base](user_defined_type const& t) -> bool {
            return t < rand_base;
        }, rand_base);

    ////////// Corner test cases.
    test_partition_copy(execution::par, IteratorTag(), int(),
        [rand_base](const int n) -> bool {
            HPX_UNUSED(rand_base);
            return true;
        }, rand_base);
    test_partition_copy(execution::par_unseq, IteratorTag(), user_defined_type(),
        [rand_base](user_defined_type const& t) -> bool {
            HPX_UNUSED(rand_base);
            return false;
        }, rand_base);
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_partition_copy_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_partition_copy_exception(execution::seq, IteratorTag());
    test_partition_copy_exception(execution::par, IteratorTag());

    test_partition_copy_exception_async(execution::seq(execution::task),
        IteratorTag());
    test_partition_copy_exception_async(execution::par(execution::task),
        IteratorTag());
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_partition_copy_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_partition_copy_bad_alloc(execution::seq, IteratorTag());
    test_partition_copy_bad_alloc(execution::par, IteratorTag());

    test_partition_copy_bad_alloc_async(execution::seq(execution::task),
        IteratorTag());
    test_partition_copy_bad_alloc_async(execution::par(execution::task),
        IteratorTag());
}

#endif

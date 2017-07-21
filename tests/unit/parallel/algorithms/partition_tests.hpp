//  Copyright (c) 2017 Taeguk Kwon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TEST_PARTITION_JUL_20_2017_0954PM)
#define HPX_PARALLEL_TEST_PARTITION_JUL_20_2017_0954PM

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

    bool operator<(user_defined_type const& t)
    {
        return this->name < t.name ||
            (this->name == t.name && this->val < t.val);
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
    random_fill(int rand_base, int half_range /* >= 0 */)
        : gen(std::rand()),
        dist(rand_base - half_range, rand_base + half_range)
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
void test_partition(ExPolicy policy, IteratorTag, DataType, Pred pred,
    std::size_t size, random_fill gen_functor)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef typename std::vector<DataType>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<DataType> c(size), c_org;
    std::generate(std::begin(c), std::end(c), gen_functor);
    c_org = c;

    auto result = hpx::parallel::partition(policy,
        iterator(std::begin(c)), iterator(std::end(c)),
        pred);

    bool is_partitioned =
        std::is_partitioned(std::begin(c), std::end(c), pred);

    HPX_TEST(is_partitioned);

    auto solution =
        std::partition_point(std::begin(c), std::end(c), pred);

    HPX_TEST(result.base() == solution);

    std::sort(std::begin(c), std::end(c));
    std::sort(std::begin(c_org), std::end(c_org));

    bool unchanged = std::equal(
        std::begin(c), std::end(c),
        std::begin(c_org), std::end(c_org));

    HPX_TEST(unchanged);
}

template <typename ExPolicy, typename IteratorTag, typename DataType, typename Pred>
void test_partition_async(ExPolicy policy, IteratorTag, DataType, Pred pred,
    std::size_t size, random_fill gen_functor)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef typename std::vector<DataType>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<DataType> c(size), c_org;
    std::generate(std::begin(c), std::end(c), gen_functor);
    c_org = c;

    auto f = hpx::parallel::partition(policy,
        iterator(std::begin(c)), iterator(std::end(c)),
        pred);
    auto result = f.get();

    bool is_partitioned =
        std::is_partitioned(std::begin(c), std::end(c), pred);

    HPX_TEST(is_partitioned);

    auto solution =
        std::partition_point(std::begin(c), std::end(c), pred);

    HPX_TEST(result.base() == solution);

    std::sort(std::begin(c), std::end(c));
    std::sort(std::begin(c_org), std::end(c_org));

    bool unchanged = std::equal(
        std::begin(c), std::end(c),
        std::begin(c_org), std::end(c_org));

    HPX_TEST(unchanged);
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_partition_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t const size = 300007;
    std::vector<int> c(size);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    try {
        auto result = hpx::parallel::partition(policy,
            iterator(std::begin(c)), iterator(std::end(c)),
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
void test_partition_exception_async(ExPolicy policy, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t const size = 300007;
    std::vector<int> c(size);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        auto f = hpx::parallel::partition(policy,
            iterator(std::begin(c)), iterator(std::end(c)),
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
void test_partition_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t const size = 300007;
    std::vector<int> c(size);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_bad_alloc = false;
    try {
        auto result = hpx::parallel::partition(policy,
            iterator(std::begin(c)), iterator(std::end(c)),
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
void test_partition_bad_alloc_async(ExPolicy policy, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t const size = 300007;
    std::vector<int> c(size);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        auto f = hpx::parallel::partition(policy,
            iterator(std::begin(c)), iterator(std::end(c)),
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

template <typename ExPolicy, typename IteratorTag, typename DataType, typename Pred>
void test_partition(ExPolicy policy, IteratorTag, DataType, Pred pred,
    int rand_base)
{
    const std::size_t size = 300007u;
    const int half_range = size / 10;

    test_partition(policy, IteratorTag(), DataType(), pred,
        size, random_fill(rand_base, half_range));
}

template <typename ExPolicy, typename IteratorTag, typename DataType, typename Pred>
void test_partition_async(ExPolicy policy, IteratorTag, DataType, Pred pred,
    int rand_base)
{
    const std::size_t size = 300007u;
    const int half_range = size / 10;

    test_partition_async(policy, IteratorTag(), DataType(), pred,
        size, random_fill(rand_base, half_range));
}

template <typename ExPolicy, typename IteratorTag, typename DataType, typename Pred>
void test_partition_heavy(ExPolicy policy, IteratorTag, DataType, Pred pred,
    int rand_base)
{
    auto size_list = {
        1, 2, 3, 4, 5, 6, 7, 8,        /* very small size */
        16, 24, 32, 48, 64,            /* intent the number of core */
        123, 4567, 65432, 123456,      /* various size */
        961230, 170228, 3456789,       /* big size */
        std::rand(), std::rand()       /* random size */
    };

    for (auto size : size_list)
    {
        auto half_range_list = {
            0, 1, size / 10
        };

        for (auto half_range : half_range_list)
        {
            test_partition(policy, IteratorTag(), DataType(), pred,
                static_cast<std::size_t>(size),
                random_fill(rand_base, half_range));
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_partition()
{
    using namespace hpx::parallel;

    int rand_base = std::rand();

    ////////// Test cases for 'int' type.
    test_partition(execution::seq, IteratorTag(), int(),
        [rand_base](const int n) -> bool {
            return n < rand_base;
        }, rand_base);
    test_partition(execution::par, IteratorTag(), int(),
        [rand_base](const int n) -> bool {
            return n <= rand_base;
        }, rand_base);
    test_partition(execution::par_unseq, IteratorTag(), int(),
        [rand_base](const int n) -> bool {
            return n > rand_base;
        }, rand_base);

    ////////// Test cases for user defined type.
    test_partition(execution::seq, IteratorTag(), user_defined_type(),
        [rand_base](user_defined_type const& t) -> bool {
            return t < rand_base;
        }, rand_base);
    test_partition(execution::par, IteratorTag(), user_defined_type(),
        [rand_base](user_defined_type const& t) -> bool {
            return !(t < rand_base);
        }, rand_base);
    test_partition(execution::par_unseq, IteratorTag(), user_defined_type(),
        [rand_base](user_defined_type const& t) -> bool {
            return t < rand_base;
        }, rand_base);

    ////////// Asynchronous test cases for 'int' type.
    test_partition_async(execution::seq(execution::task), IteratorTag(), int(),
        [rand_base](const int n) -> bool {
            return n >= rand_base;
        }, rand_base);
    test_partition_async(execution::par(execution::task), IteratorTag(), int(),
        [rand_base](const int n) -> bool {
            return n < rand_base;
        }, rand_base);

    ////////// Asynchronous test cases for user defined type.
    test_partition_async(execution::seq(execution::task), IteratorTag(),
        user_defined_type(),
        [rand_base](user_defined_type const& t) -> bool {
            return !(t < rand_base);
        }, rand_base);
    test_partition_async(execution::par(execution::task), IteratorTag(),
        user_defined_type(),
        [rand_base](user_defined_type const& t) -> bool {
            return t < rand_base;
        }, rand_base);

    ////////// Corner test cases.
    test_partition(execution::par, IteratorTag(), int(),
        [](const int n) -> bool {
            return true;
        }, rand_base);
    test_partition(execution::par_unseq, IteratorTag(), user_defined_type(),
        [](user_defined_type const& t) -> bool {
            return false;
        }, rand_base);

    ////////// Many test cases for meticulous tests.
    test_partition_heavy(execution::par, IteratorTag(), int(),
        [rand_base](const int n) -> bool {
            return n < rand_base;
        }, rand_base);
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_partition_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_partition_exception(execution::seq, IteratorTag());
    test_partition_exception(execution::par, IteratorTag());

    test_partition_exception_async(execution::seq(execution::task),
        IteratorTag());
    test_partition_exception_async(execution::par(execution::task),
        IteratorTag());
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_partition_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_partition_bad_alloc(execution::seq, IteratorTag());
    test_partition_bad_alloc(execution::par, IteratorTag());

    test_partition_bad_alloc_async(execution::seq(execution::task),
        IteratorTag());
    test_partition_bad_alloc_async(execution::par(execution::task),
        IteratorTag());
}

#endif

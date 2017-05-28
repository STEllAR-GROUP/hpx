//  Copyright (c) 2017 Taeguk Kwon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TEST_is_heap_until_JUN_28_2017_1745PM)
#define HPX_PARALLEL_TEST_is_heap_until_JUN_28_2017_1745PM

#include <hpx/include/parallel_is_heap.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include <cstddef>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
struct throw_always
{
    template <typename T1, typename T2>
    bool operator()(T1 const&, T2 const&) const
    {
        throw std::runtime_error("test");
    }
};

struct throw_bad_alloc
{
    template <typename T1, typename T2>
    bool operator()(T1 const&, T2 const&) const
    {
        throw std::bad_alloc();
    }
}; 

struct user_defined_type
{
    user_defined_type() = default;
    user_defined_type(int rand_no) : val(rand_no) {}

    bool operator<(user_defined_type const& t) const
    {
        if (this->name < t.name)
            return true;
        else if (this->name < t.name)
            return false;
        else
            return this->val < t.val;
    }

    const user_defined_type& operator++()
    {
        static const std::vector<std::string> name_list = {
            "ABB", "ABC", "ACB", "BCA", "CAA", "CAAA", "CAAB"
        };
        name = name_list[std::rand() % name_list.size()];
        ++val;
        return *this;
    }

    std::string name;
    int val;
};

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag, typename DataType = int>
void test_is_heap_until(ExPolicy policy, IteratorTag, DataType = DataType())
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<DataType>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<DataType> c(10007);
    std::iota(boost::begin(c), boost::end(c), DataType(std::rand()));

    auto heap_end_iter = std::next(boost::begin(c), std::rand() % c.size());
    std::make_heap(boost::begin(c), heap_end_iter);

    iterator result =
        hpx::parallel::is_heap_until(policy,
            iterator(boost::begin(c)), iterator(boost::end(c)));

    auto solution = std::is_heap_until(std::begin(c), std::end(c));

    HPX_TEST(result.base() == solution);
}

template <typename ExPolicy, typename IteratorTag, typename DataType, typename Pred>
void test_is_heap_until(ExPolicy policy, IteratorTag, DataType, Pred pred)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<DataType>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<DataType> c(10007);
    std::iota(boost::begin(c), boost::end(c), DataType(std::rand()));

    auto heap_end_iter = std::next(boost::begin(c), std::rand() % c.size());
    std::make_heap(boost::begin(c), heap_end_iter);

    iterator result =
        hpx::parallel::is_heap_until(policy,
            iterator(boost::begin(c)), iterator(boost::end(c)), pred);

    auto solution = std::is_heap_until(std::begin(c), std::end(c), pred);

    HPX_TEST(result.base() == solution);
}

template <typename ExPolicy, typename IteratorTag, typename DataType = int>
void test_is_heap_until_async(ExPolicy policy, IteratorTag, DataType = DataType())
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<DataType>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<DataType> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    auto heap_end_iter = std::next(boost::begin(c), std::rand() % c.size());
    std::make_heap(boost::begin(c), heap_end_iter);

    auto f =
        hpx::parallel::is_heap_until(policy,
            iterator(boost::begin(c)), iterator(boost::end(c)));

    iterator result = f.get();
    auto solution = std::is_heap_until(std::begin(c), std::end(c));

    HPX_TEST(result.base() == solution);
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_is_heap_until_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());
    std::make_heap(boost::begin(c), boost::end(c));

    bool caught_exception = false;
    try {
        hpx::parallel::is_heap_until(policy,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            throw_always());

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
void test_is_heap_until_exception_async(ExPolicy policy, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());
    std::make_heap(boost::begin(c), boost::end(c));

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        auto f =
            hpx::parallel::is_heap_until(policy,
                iterator(boost::begin(c)), iterator(boost::end(c)),
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
void test_is_heap_until_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());
    std::make_heap(boost::begin(c), boost::end(c));

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::is_heap_until(policy,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            throw_bad_alloc());

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
void test_is_heap_until_bad_alloc_async(ExPolicy policy, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());
    std::make_heap(boost::begin(c), boost::end(c));

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        auto f =
            hpx::parallel::is_heap_until(policy,
                iterator(boost::begin(c)), iterator(boost::end(c)),
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

#endif

//  Copyright (c) 2017 Taeguk Kwon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_test_is_heap_JUN_28_2017_1745PM)
#define HPX_PARALLEL_test_is_heap_JUN_28_2017_1745PM

#include <hpx/include/parallel_is_heap.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
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
        else if (this->name > t.name)
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
void test_is_heap(ExPolicy policy, IteratorTag, DataType = DataType(),
    bool test_for_is_heap = true)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef typename std::vector<DataType>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<DataType> c(10007);
    std::iota(std::begin(c), std::end(c), DataType(std::rand()));

    auto heap_end_iter = std::next(std::begin(c), std::rand() % c.size());
    std::make_heap(std::begin(c), heap_end_iter);

    if (test_for_is_heap)
    {
        bool result = hpx::parallel::is_heap(policy,
            iterator(std::begin(c)), iterator(std::end(c)));
        bool solution = std::is_heap(std::begin(c), std::end(c));

        HPX_TEST(result == solution);
    }
    else
    {
        iterator result = hpx::parallel::is_heap_until(policy,
            iterator(std::begin(c)), iterator(std::end(c)));
        auto solution = std::is_heap_until(std::begin(c), std::end(c));

        HPX_TEST(result.base() == solution);
    }
}

template <typename ExPolicy, typename IteratorTag, typename DataType, typename Pred>
void test_is_heap_with_pred(ExPolicy policy, IteratorTag, DataType, Pred pred,
    bool test_for_is_heap = true)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef typename std::vector<DataType>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<DataType> c(10007);
    std::iota(std::begin(c), std::end(c), DataType(std::rand()));

    auto heap_end_iter = std::next(std::begin(c), std::rand() % c.size());
    std::make_heap(std::begin(c), heap_end_iter);

    if (test_for_is_heap)
    {
        bool result = hpx::parallel::is_heap(policy,
            iterator(std::begin(c)), iterator(std::end(c)), pred);
        bool solution = std::is_heap(std::begin(c), std::end(c), pred);

        HPX_TEST(result == solution);
    }
    else
    {
        iterator result = hpx::parallel::is_heap_until(policy,
            iterator(std::begin(c)), iterator(std::end(c)), pred);
        auto solution = std::is_heap_until(std::begin(c), std::end(c), pred);

        HPX_TEST(result.base() == solution);
    }
}

template <typename ExPolicy, typename IteratorTag, typename DataType = int>
void test_is_heap_async(ExPolicy policy, IteratorTag, DataType = DataType(),
    bool test_for_is_heap = true)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef typename std::vector<DataType>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<DataType> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    auto heap_end_iter = std::next(std::begin(c), std::rand() % c.size());
    std::make_heap(std::begin(c), heap_end_iter);

    if (test_for_is_heap)
    {
        auto f = hpx::parallel::is_heap(policy,
            iterator(std::begin(c)), iterator(std::end(c)));
        bool result = f.get();
        bool solution = std::is_heap(std::begin(c), std::end(c));

        HPX_TEST(result == solution);
    }
    else
    {
        auto f = hpx::parallel::is_heap_until(policy,
            iterator(std::begin(c)), iterator(std::end(c)));
        iterator result = f.get();
        auto solution = std::is_heap_until(std::begin(c), std::end(c));

        HPX_TEST(result.base() == solution);
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_is_heap_exception(ExPolicy policy, IteratorTag,
    bool test_for_is_heap = true)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());
    std::make_heap(std::begin(c), std::end(c));

    bool caught_exception = false;
    try {
        if (test_for_is_heap)
        {
            hpx::parallel::is_heap(policy,
                iterator(std::begin(c)), iterator(std::end(c)),
                throw_always());
        }
        else
        {
            hpx::parallel::is_heap_until(policy,
                iterator(std::begin(c)), iterator(std::end(c)),
                throw_always());
        }

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
void test_is_heap_exception_async(ExPolicy policy, IteratorTag,
    bool test_for_is_heap = true)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());
    std::make_heap(std::begin(c), std::end(c));

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        if (test_for_is_heap)
        {
            auto f = hpx::parallel::is_heap(policy,
                iterator(std::begin(c)), iterator(std::end(c)),
                throw_always());
            returned_from_algorithm = true;
            f.get();
        }
        else
        {
            auto f = hpx::parallel::is_heap_until(policy,
                iterator(std::begin(c)), iterator(std::end(c)),
                throw_always());
            returned_from_algorithm = true;
            f.get();
        }

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
void test_is_heap_bad_alloc(ExPolicy policy, IteratorTag,
    bool test_for_is_heap = true)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());
    std::make_heap(std::begin(c), std::end(c));

    bool caught_bad_alloc = false;
    try {
        if (test_for_is_heap)
        {
            hpx::parallel::is_heap(policy,
                iterator(std::begin(c)), iterator(std::end(c)),
                throw_bad_alloc());
        }
        else
        {
            hpx::parallel::is_heap_until(policy,
                iterator(std::begin(c)), iterator(std::end(c)),
                throw_bad_alloc());
        }

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
void test_is_heap_bad_alloc_async(ExPolicy policy, IteratorTag,
    bool test_for_is_heap = true)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());
    std::make_heap(std::begin(c), std::end(c));

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        if (test_for_is_heap)
        {
            auto f = hpx::parallel::is_heap(policy,
                iterator(std::begin(c)), iterator(std::end(c)),
                throw_bad_alloc());
            returned_from_algorithm = true;
            f.get();
        }
        else
        {
            auto f = hpx::parallel::is_heap_until(policy,
                iterator(std::begin(c)), iterator(std::end(c)),
                throw_bad_alloc());
            returned_from_algorithm = true;
            f.get();
        }

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
void test_is_heap(bool test_for_is_heap = true)
{
    using namespace hpx::parallel;

    test_is_heap(execution::seq, IteratorTag(), test_for_is_heap);
    test_is_heap(execution::par, IteratorTag(), test_for_is_heap);
    test_is_heap(execution::par_unseq, IteratorTag(), test_for_is_heap);

    test_is_heap(execution::seq, IteratorTag(), user_defined_type(),
        test_for_is_heap);
    test_is_heap(execution::par, IteratorTag(), user_defined_type(),
        test_for_is_heap);
    test_is_heap(execution::par_unseq, IteratorTag(), user_defined_type(),
        test_for_is_heap);

    test_is_heap_with_pred(execution::seq, IteratorTag(), int(),
        std::greater<int>(), test_for_is_heap);
    test_is_heap_with_pred(execution::par, IteratorTag(), int(),
        std::less<int>(), test_for_is_heap);
    test_is_heap_with_pred(execution::par_unseq, IteratorTag(), int(),
        std::greater_equal<int>(), test_for_is_heap);

    test_is_heap_async(execution::seq(execution::task), IteratorTag(),
        test_for_is_heap);
    test_is_heap_async(execution::par(execution::task), IteratorTag(),
        test_for_is_heap);

    test_is_heap_async(execution::seq(execution::task), IteratorTag(),
        user_defined_type(), test_for_is_heap);
    test_is_heap_async(execution::par(execution::task), IteratorTag(),
        user_defined_type(), test_for_is_heap);
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_is_heap_exception(bool test_for_is_heap = true)
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_is_heap_exception(execution::seq, IteratorTag(), test_for_is_heap);
    test_is_heap_exception(execution::par, IteratorTag(), test_for_is_heap);

    test_is_heap_exception_async(execution::seq(execution::task), IteratorTag(),
        test_for_is_heap);
    test_is_heap_exception_async(execution::par(execution::task), IteratorTag(),
        test_for_is_heap);
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_is_heap_bad_alloc(bool test_for_is_heap = true)
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_is_heap_bad_alloc(execution::seq, IteratorTag(), test_for_is_heap);
    test_is_heap_bad_alloc(execution::par, IteratorTag(), test_for_is_heap);

    test_is_heap_bad_alloc_async(execution::seq(execution::task), IteratorTag(),
        test_for_is_heap);
    test_is_heap_bad_alloc_async(execution::par(execution::task), IteratorTag(),
        test_for_is_heap);
}

#endif

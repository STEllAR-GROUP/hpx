//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TEST_UNINIT_COPY_MAY28_15_1344)
#define HPX_PARALLEL_TEST_UNINIT_COPY_MAY28_15_1344

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_uninitialized_copy.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>
#include <boost/range/functions.hpp>

#include <cstddef>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_copy(ExPolicy && policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(boost::begin(c), boost::end(c), std::rand());
    hpx::parallel::uninitialized_copy(std::forward<ExPolicy>(policy),
        iterator(boost::begin(c)), iterator(boost::end(c)), boost::begin(d));

    std::size_t count = 0;
    HPX_TEST(std::equal(boost::begin(c), boost::end(c), boost::begin(d),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_copy_async(ExPolicy && p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(boost::begin(c), boost::end(c), std::rand());

    hpx::future<base_iterator> f =
        hpx::parallel::uninitialized_copy(std::forward<ExPolicy>(p),
            iterator(boost::begin(c)), iterator(boost::end(c)),
            boost::begin(d));
    f.wait();

    std::size_t count = 0;
    HPX_TEST(std::equal(boost::begin(c), boost::end(c), boost::begin(d),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d.size());
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_copy_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<test::count_instances>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<test::count_instances> c(10007);
    std::vector<test::count_instances> d(c.size());
    std::iota(boost::begin(c), boost::end(c), std::rand());

    boost::atomic<std::size_t> throw_after(std::rand() % c.size()); //-V104
    test::count_instances::instance_count.store(0);

    bool caught_exception = false;
    try {
        hpx::parallel::uninitialized_copy(policy,
            decorated_iterator(
                boost::begin(c),
                [&throw_after]()
                {
                    if (throw_after-- == 0)
                        throw std::runtime_error("test");
                }),
            decorated_iterator(boost::end(c)),
            boost::begin(d));
        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(policy, e);
    }
    catch (...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST_EQ(test::count_instances::instance_count.load(), std::size_t(0));
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_copy_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<test::count_instances>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<test::count_instances> c(10007);
    std::vector<test::count_instances> d(c.size());
    std::iota(boost::begin(c), boost::end(c), std::rand());

    boost::atomic<std::size_t> throw_after(std::rand() % c.size()); //-V104
    test::count_instances::instance_count.store(0);

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<base_iterator> f =
            hpx::parallel::uninitialized_copy(p,
                decorated_iterator(
                    boost::begin(c),
                    [&throw_after]()
                    {
                        if (throw_after-- == 0)
                            throw std::runtime_error("test");
                    }),
                decorated_iterator(boost::end(c)),
                boost::begin(d));

        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(p, e);
    }
    catch (...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST(returned_from_algorithm);
    HPX_TEST_EQ(test::count_instances::instance_count.load(), std::size_t(0));
}

//////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_copy_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<test::count_instances>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<test::count_instances> c(10007);
    std::vector<test::count_instances> d(c.size());
    std::iota(boost::begin(c), boost::end(c), std::rand());

    boost::atomic<std::size_t> throw_after(std::rand() % c.size()); //-V104
    test::count_instances::instance_count.store(0);

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::uninitialized_copy(policy,
            decorated_iterator(
                boost::begin(c),
                [&throw_after]()
                {
                    if (throw_after-- == 0)
                        throw std::bad_alloc();
                }),
            decorated_iterator(boost::end(c)),
            boost::begin(d));

        HPX_TEST(false);
    }
    catch (std::bad_alloc const&) {
        caught_bad_alloc = true;
    }
    catch (...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
    HPX_TEST_EQ(test::count_instances::instance_count.load(), std::size_t(0));
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_copy_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<test::count_instances>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<test::count_instances> c(10007);
    std::vector<test::count_instances> d(c.size());
    std::iota(boost::begin(c), boost::end(c), std::rand());

    boost::atomic<std::size_t> throw_after(std::rand() % c.size()); //-V104
    test::count_instances::instance_count.store(0);

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<base_iterator> f =
            hpx::parallel::uninitialized_copy(p,
                decorated_iterator(
                    boost::begin(c),
                    [&throw_after]()
                    {
                        if (throw_after-- == 0)
                            throw std::bad_alloc();
                    }),
                decorated_iterator(boost::end(c)),
                boost::begin(d));

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
    HPX_TEST_EQ(test::count_instances::instance_count.load(), std::size_t(0));
}

#endif

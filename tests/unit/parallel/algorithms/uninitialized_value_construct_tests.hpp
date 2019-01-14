//  Copyright (c) 2014-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TEST_UNINIT_COPY_MAY28_15_1344)
#define HPX_PARALLEL_TEST_UNINIT_COPY_MAY28_15_1344

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/parallel_uninitialized_value_construct.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <atomic>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>

#include "test_utils.hpp"

struct value_constructable
{
    int value_;
};

std::size_t const data_size = 10007;

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_value_construct(ExPolicy && policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef value_constructable* base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    value_constructable* p = (value_constructable*)std::malloc(
        data_size * sizeof(value_constructable));
    std::memset(
        static_cast<void*>(p), 0xcd, data_size * sizeof(value_constructable));

    hpx::parallel::uninitialized_value_construct(
        std::forward<ExPolicy>(policy),
        iterator(p), iterator(p + data_size));

    std::size_t count = 0;
    std::for_each(p, p + data_size,
        [&count](value_constructable v1)
        {
            HPX_TEST_EQ(v1.value_, 0);
            ++count;
        });
    HPX_TEST_EQ(count, data_size);

    std::free(p);
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_value_construct_async(ExPolicy && policy, IteratorTag)
{
    typedef value_constructable* base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    value_constructable* p = (value_constructable*)std::malloc(
        data_size * sizeof(value_constructable));
    std::memset(
        static_cast<void*>(p), 0xcd, data_size * sizeof(value_constructable));

    auto f =
        hpx::parallel::uninitialized_value_construct(
            std::forward<ExPolicy>(policy),
            iterator(p), iterator(p + data_size));
    f.wait();

    std::size_t count = 0;
    std::for_each(p, p + data_size,
        [&count](value_constructable v1)
        {
            HPX_TEST_EQ(v1.value_, 0);
            ++count;
        });
    HPX_TEST_EQ(count, data_size);

    std::free(p);
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_value_construct_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef test::count_instances_v<value_constructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*)std::malloc(data_size * sizeof(data_type));
    std::memset(static_cast<void*>(p), 0xcd, data_size * sizeof(data_type));

    std::atomic<std::size_t> throw_after(std::rand() % data_size); //-V104
    std::size_t throw_after_ = throw_after.load();

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    bool caught_exception = false;
    try {
        hpx::parallel::uninitialized_value_construct(policy,
            decorated_iterator(p,
                [&throw_after]()
                {
                    if (throw_after-- == 0)
                        throw std::runtime_error("test");
                }),
            decorated_iterator(p + data_size));
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
    HPX_TEST_EQ(data_type::instance_count.load(), std::size_t(0));
    HPX_TEST_LTE(throw_after_, data_type::max_instance_count.load());

    std::free(p);
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_value_construct_exception_async(
    ExPolicy policy, IteratorTag)
{
    typedef test::count_instances_v<value_constructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*)std::malloc(data_size * sizeof(data_type));
    std::memset(static_cast<void*>(p), 0xcd, data_size * sizeof(data_type));

    std::atomic<std::size_t> throw_after(std::rand() % data_size); //-V104
    std::size_t throw_after_ = throw_after.load();

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        auto f =
            hpx::parallel::uninitialized_value_construct(policy,
                decorated_iterator(
                    p,
                    [&throw_after]()
                    {
                        if (throw_after-- == 0)
                            throw std::runtime_error("test");
                    }),
                decorated_iterator(p + data_size));

        returned_from_algorithm = true;
        f.get();

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
    HPX_TEST(returned_from_algorithm);
    HPX_TEST_EQ(data_type::instance_count.load(), std::size_t(0));
    HPX_TEST_LTE(throw_after_, data_type::max_instance_count.load());

    std::free(p);
}

//////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_value_construct_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef test::count_instances_v<value_constructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*)std::malloc(data_size * sizeof(data_type));
    std::memset(static_cast<void*>(p), 0xcd, data_size * sizeof(data_type));

    std::atomic<std::size_t> throw_after(std::rand() % data_size); //-V104
    std::size_t throw_after_ = throw_after.load();

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::uninitialized_value_construct(policy,
            decorated_iterator(
                p,
                [&throw_after]()
                {
                    if (throw_after-- == 0)
                        throw std::bad_alloc();
                }),
            decorated_iterator(p + data_size));

        HPX_TEST(false);
    }
    catch (std::bad_alloc const&) {
        caught_bad_alloc = true;
    }
    catch (...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
    HPX_TEST_EQ(data_type::instance_count.load(), std::size_t(0));
    HPX_TEST_LTE(throw_after_, data_type::max_instance_count.load());

    std::free(p);
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_value_construct_bad_alloc_async(
    ExPolicy policy, IteratorTag)
{
    typedef test::count_instances_v<value_constructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*)std::malloc(data_size * sizeof(data_type));
    std::memset(static_cast<void*>(p), 0xcd, data_size * sizeof(data_type));

    std::atomic<std::size_t> throw_after(std::rand() % data_size); //-V104
    std::size_t throw_after_ = throw_after.load();

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        auto f =
            hpx::parallel::uninitialized_value_construct(policy,
                decorated_iterator(
                    p,
                    [&throw_after]()
                    {
                        if (throw_after-- == 0)
                            throw std::bad_alloc();
                    }),
                decorated_iterator(p + data_size));

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
    HPX_TEST_EQ(data_type::instance_count.load(), std::size_t(0));
    HPX_TEST_LTE(throw_after_, data_type::max_instance_count.load());

    std::free(p);
}

#endif

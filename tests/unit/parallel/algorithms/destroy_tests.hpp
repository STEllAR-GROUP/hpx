//  Copyright (c) 2014-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TEST_UNINIT_COPY_MAY28_15_1344)
#define HPX_PARALLEL_TEST_UNINIT_COPY_MAY28_15_1344

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/parallel_destroy.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>
#include <boost/range/functions.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>

#include "test_utils.hpp"

struct destructable
{
    destructable()
      : value_(0)
    {}

    ~destructable()
    {
        std::memset(&value_, 0xcd, sizeof(value_));
    }

    std::uint32_t value_;
};

std::size_t const data_size = 10007;

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_destroy(ExPolicy && policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef destructable* base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    destructable* p = (destructable*)std::malloc(
        data_size * sizeof(destructable));

    // value-initialize data in array
    std::for_each(
        p, p + data_size,
        [](destructable& d)
        {
            ::new (static_cast<void*>(std::addressof(d))) destructable;
        });

    hpx::parallel::destroy(
        std::forward<ExPolicy>(policy),
        iterator(p), iterator(p + data_size));

    std::size_t count = 0;
    std::for_each(p, p + data_size,
        [&count](destructable v1)
        {
            HPX_TEST_EQ(v1.value_, (std::uint32_t)0xcdcdcdcd);
            ++count;
        });
    HPX_TEST_EQ(count, data_size);

    std::free(p);
}

template <typename ExPolicy, typename IteratorTag>
void test_destroy_async(ExPolicy && policy, IteratorTag)
{
    typedef destructable* base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    destructable* p = (destructable*)std::malloc(
        data_size * sizeof(destructable));

    // value-initialize data in array
    std::for_each(
        p, p + data_size,
        [](destructable& d)
        {
            ::new (static_cast<void*>(std::addressof(d))) destructable;
        });

    auto f =
        hpx::parallel::destroy(
            std::forward<ExPolicy>(policy),
            iterator(p), iterator(p + data_size));
    f.wait();

    std::size_t count = 0;
    std::for_each(p, p + data_size,
        [&count](destructable v1)
        {
            HPX_TEST_EQ(v1.value_, (std::uint32_t)0xcdcdcdcd);
            ++count;
        });
    HPX_TEST_EQ(count, data_size);

    std::free(p);
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_destroy_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef test::count_instances_v<destructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*)std::malloc(data_size * sizeof(data_type));

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    // value-initialize data in array
    std::for_each(
        p, p + data_size,
        [](data_type& d)
        {
            ::new (static_cast<void*>(std::addressof(d))) data_type;
        });

    HPX_TEST_EQ(data_type::instance_count.load(), data_size);

    boost::atomic<std::size_t> throw_after(std::rand() % data_size); //-V104
    std::int64_t throw_after_ = throw_after.load();

    bool caught_exception = false;
    try {
        hpx::parallel::destroy(policy,
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
    HPX_TEST_LTE(data_type::instance_count.load(),
        std::size_t(data_size-throw_after_));

    std::free(p);
}

template <typename ExPolicy, typename IteratorTag>
void test_destroy_exception_async(
    ExPolicy policy, IteratorTag)
{
    typedef test::count_instances_v<destructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*)std::malloc(data_size * sizeof(data_type));

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    // value-initialize data in array
    std::for_each(
        p, p + data_size,
        [](data_type& d)
        {
            ::new (static_cast<void*>(std::addressof(d))) data_type;
        });

    HPX_TEST_EQ(data_type::instance_count.load(), data_size);

    boost::atomic<std::size_t> throw_after(std::rand() % data_size); //-V104
    std::int64_t throw_after_ = throw_after.load();

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        auto f =
            hpx::parallel::destroy(policy,
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
    HPX_TEST_LTE(data_type::instance_count.load(),
        std::size_t(data_size-throw_after_));

    std::free(p);
}

//////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_destroy_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef test::count_instances_v<destructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*)std::malloc(data_size * sizeof(data_type));

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    // value-initialize data in array
    std::for_each(
        p, p + data_size,
        [](data_type& d)
        {
            ::new (static_cast<void*>(std::addressof(d))) data_type;
        });

    HPX_TEST_EQ(data_type::instance_count.load(), data_size);

    boost::atomic<std::size_t> throw_after(std::rand() % data_size); //-V104
    std::int64_t throw_after_ = throw_after.load();

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::destroy(policy,
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
    HPX_TEST_LTE(data_type::instance_count.load(),
        std::size_t(data_size-throw_after_));

    std::free(p);
}

template <typename ExPolicy, typename IteratorTag>
void test_destroy_bad_alloc_async(
    ExPolicy policy, IteratorTag)
{
    typedef test::count_instances_v<destructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*)std::malloc(data_size * sizeof(data_type));

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    // value-initialize data in array
    std::for_each(
        p, p + data_size,
        [](data_type& d)
        {
            ::new (static_cast<void*>(std::addressof(d))) data_type;
        });

    HPX_TEST_EQ(data_type::instance_count.load(), data_size);

    boost::atomic<std::size_t> throw_after(std::rand() % data_size); //-V104
    std::int64_t throw_after_ = throw_after.load();

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        auto f =
            hpx::parallel::destroy(policy,
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
    HPX_TEST_LTE(data_type::instance_count.load(),
        std::size_t(data_size-throw_after_));

    std::free(p);
}

#endif

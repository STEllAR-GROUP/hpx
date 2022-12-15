//  Copyright (C) 2011 Tim Blechmann
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/concurrency.hpp>
#include <hpx/modules/testing.hpp>

#include <memory>

#include "test_helpers.hpp"

void simple_queue_test()
{
    hpx::lockfree::queue<int> f(64);

    HPX_TEST(f.is_lock_free());

    HPX_TEST(f.empty());
    f.push(1);
    f.push(2);

    int i1(0), i2(0);

    HPX_TEST(f.pop(i1));
    HPX_TEST_EQ(i1, 1);

    HPX_TEST(f.pop(i2));
    HPX_TEST_EQ(i2, 2);
    HPX_TEST(f.empty());
}

void simple_queue_test_capacity()
{
    hpx::lockfree::queue<int, std::allocator<int>, 64> f;

    HPX_TEST(f.empty());
    f.push(1);
    f.push(2);

    int i1(0), i2(0);

    HPX_TEST(f.pop(i1));
    HPX_TEST_EQ(i1, 1);

    HPX_TEST(f.pop(i2));
    HPX_TEST_EQ(i2, 2);
    HPX_TEST(f.empty());
}

void unsafe_queue_test()
{
    hpx::lockfree::queue<int> f(64);

    HPX_TEST(f.is_lock_free());
    HPX_TEST(f.empty());

    int i1(0), i2(0);

    f.unsynchronized_push(1);
    f.unsynchronized_push(2);

    HPX_TEST(f.unsynchronized_pop(i1));
    HPX_TEST_EQ(i1, 1);

    HPX_TEST(f.unsynchronized_pop(i2));
    HPX_TEST_EQ(i2, 2);
    HPX_TEST(f.empty());
}

void queue_consume_one_test()
{
    hpx::lockfree::queue<int> f(64);

    HPX_TEST(f.is_lock_free());
    HPX_TEST(f.empty());

    f.push(1);
    f.push(2);

    bool success1 = f.consume_one([](int i) { HPX_TEST_EQ(i, 1); });
    bool success2 = f.consume_one([](int i) { HPX_TEST_EQ(i, 2); });

    HPX_TEST(success1);
    HPX_TEST(success2);

    HPX_TEST(f.empty());
}

void queue_consume_all_test()
{
    hpx::lockfree::queue<int> f(64);

    HPX_TEST(f.is_lock_free());
    HPX_TEST(f.empty());

    f.push(1);
    f.push(2);

    size_t consumed = f.consume_all([](int) {});

    HPX_TEST_EQ(consumed, 2u);

    HPX_TEST(f.empty());
}

void queue_convert_pop_test()
{
    hpx::lockfree::queue<int*> f(128);
    HPX_TEST(f.empty());
    f.push(new int(1));
    f.push(new int(2));
    f.push(new int(3));
    f.push(new int(4));

    {
        int* i1;

        HPX_TEST(f.pop(i1));
        HPX_TEST_EQ(*i1, 1);
        delete i1;
    }

    {
        std::shared_ptr<int> i2;
        HPX_TEST(f.pop(i2));
        HPX_TEST_EQ(*i2, 2);
    }

    {
        std::unique_ptr<int> i3;
        HPX_TEST(f.pop(i3));
        HPX_TEST_EQ(*i3, 3);
    }

    {
        std::shared_ptr<int> i4;
        HPX_TEST(f.pop(i4));
        HPX_TEST_EQ(*i4, 4);
    }

    HPX_TEST(f.empty());
}

void reserve_test()
{
    hpx::lockfree::queue<void*> ms(1);
    ms.reserve(1);
    ms.reserve_unsafe(1);
}

int main()
{
    simple_queue_test();
    simple_queue_test_capacity();
    unsafe_queue_test();
    queue_consume_one_test();
    queue_consume_all_test();
    queue_convert_pop_test();
    reserve_test();

    return hpx::util::report_errors();
}

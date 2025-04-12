//  Copyright (C) 2011 Tim Blechmann
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/concurrency.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <memory>

#include "test_helpers.hpp"

void simple_stack_test()
{
    hpx::lockfree::stack<long> stk(128);

    stk.push(1);
    stk.push(2);
    long out;
    HPX_TEST(stk.pop(out));
    HPX_TEST_EQ(out, 2);
    HPX_TEST(stk.pop(out));
    HPX_TEST_EQ(out, 1);
    HPX_TEST(!stk.pop(out));
}

void unsafe_stack_test()
{
    hpx::lockfree::stack<long> stk(128);

    stk.unsynchronized_push(1);
    stk.unsynchronized_push(2);
    long out;
    HPX_TEST(stk.unsynchronized_pop(out));
    HPX_TEST_EQ(out, 2);
    HPX_TEST(stk.unsynchronized_pop(out));
    HPX_TEST_EQ(out, 1);
    HPX_TEST(!stk.unsynchronized_pop(out));
}

void ranged_push_test()
{
    hpx::lockfree::stack<long> stk(128);

    long data[2] = {1, 2};

    HPX_TEST_EQ(stk.push(data, data + 2), data + 2);

    long out;
    HPX_TEST(stk.unsynchronized_pop(out));
    HPX_TEST_EQ(out, 2);
    HPX_TEST(stk.unsynchronized_pop(out));
    HPX_TEST_EQ(out, 1);
    HPX_TEST(!stk.unsynchronized_pop(out));
}

void ranged_unsynchronized_push_test()
{
    hpx::lockfree::stack<long> stk(128);

    long data[2] = {1, 2};

    HPX_TEST_EQ(stk.unsynchronized_push(data, data + 2), data + 2);

    long out;
    HPX_TEST(stk.unsynchronized_pop(out));
    HPX_TEST_EQ(out, 2);
    HPX_TEST(stk.unsynchronized_pop(out));
    HPX_TEST_EQ(out, 1);
    HPX_TEST(!stk.unsynchronized_pop(out));
}

void variable_size_stack_test()
{
    hpx::lockfree::stack<long, std::allocator<long>, 128> stk;

    stk.push(1);
    stk.push(2);

    long out;
    HPX_TEST(stk.pop(out));
    HPX_TEST_EQ(out, 2);
    HPX_TEST(stk.pop(out));
    HPX_TEST_EQ(out, 1);
    HPX_TEST(!stk.pop(out));
    HPX_TEST(stk.empty());
}

void variable_size_stack_test_exhausted()
{
    hpx::lockfree::stack<long, std::allocator<long>, 2> stk;

    stk.push(1);
    stk.push(2);
    HPX_TEST(!stk.push(3));
    long out;
    HPX_TEST(stk.pop(out));
    HPX_TEST_EQ(out, 2);
    HPX_TEST(stk.pop(out));
    HPX_TEST_EQ(out, 1);
    HPX_TEST(!stk.pop(out));
    HPX_TEST(stk.empty());
}

void bounded_stack_test_exhausted()
{
    hpx::lockfree::stack<long> stk(2);

    stk.bounded_push(1);
    stk.bounded_push(2);
    HPX_TEST(!stk.bounded_push(3));
    long out;
    HPX_TEST(stk.pop(out));
    HPX_TEST_EQ(out, 2);
    HPX_TEST(stk.pop(out));
    HPX_TEST_EQ(out, 1);
    HPX_TEST(!stk.pop(out));
    HPX_TEST(stk.empty());
}

void stack_consume_one_test()
{
    hpx::lockfree::stack<int> f(64);

    HPX_TEST(f.is_lock_free());
    HPX_TEST(f.empty());

    f.push(1);
    f.push(2);

    bool success1 = f.consume_one([](int i) { HPX_TEST_EQ(i, 2); });
    bool success2 = f.consume_one([](int i) { HPX_TEST_EQ(i, 1); });

    HPX_TEST(success1);
    HPX_TEST(success2);

    HPX_TEST(f.empty());
}

void stack_consume_all_test()
{
    hpx::lockfree::stack<int> f(64);

    HPX_TEST(f.is_lock_free());
    HPX_TEST(f.empty());

    f.push(1);
    f.push(2);

    std::size_t consumed = f.consume_all([](int) {});

    HPX_TEST_EQ(consumed, 2u);

    HPX_TEST(f.empty());
}

void stack_consume_all_atomic_test()
{
    hpx::lockfree::stack<int> f(64);

    HPX_TEST(f.is_lock_free());
    HPX_TEST(f.empty());

    f.push(1);
    f.push(2);
    f.push(3);

    std::size_t consumed = f.consume_all_atomic([](int) {});

    HPX_TEST_EQ(consumed, 3u);

    HPX_TEST(f.empty());
}

void stack_consume_all_atomic_reversed_test()
{
    hpx::lockfree::stack<int> f(64);

    HPX_TEST(f.is_lock_free());
    HPX_TEST(f.empty());

    f.push(1);
    f.push(2);
    f.push(3);

    std::size_t consumed = f.consume_all_atomic_reversed([](int) {});

    HPX_TEST_EQ(consumed, 3u);

    HPX_TEST(f.empty());
}

void reserve_test()
{
    hpx::lockfree::stack<void*> ms(1);
    ms.reserve(1);
    ms.reserve_unsafe(1);
}

int main()
{
    simple_stack_test();
    unsafe_stack_test();
    ranged_push_test();
    ranged_unsynchronized_push_test();
    variable_size_stack_test();
    variable_size_stack_test_exhausted();
    bounded_stack_test_exhausted();
    stack_consume_one_test();
    stack_consume_all_test();
    stack_consume_all_atomic_test();
    stack_consume_all_atomic_reversed_test();
    reserve_test();

    return hpx::util::report_errors();
}

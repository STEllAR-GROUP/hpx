//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  This work is inspired by https://github.com/aprell/tasking-2.0

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/synchronization/channel_spsc.hpp>

#include <hpx/modules/testing.hpp>

#include <functional>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int verify_fibonacci(int n)
{
    if (n < 2)
        return n;
    return verify_fibonacci(n - 1) + verify_fibonacci(n - 2);
}

///////////////////////////////////////////////////////////////////////////////
template <typename T>
inline T channel_get(hpx::lcos::local::channel_spsc<T> const& c)
{
    T result;
    while (!c.get(&result))
    {
        hpx::this_thread::yield();
    }
    return result;
}

template <typename T>
inline void channel_set(hpx::lcos::local::channel_spsc<T>& c, T val)
{
    while (!c.set(std::move(val)))    // NOLINT
    {
        hpx::this_thread::yield();
    }
}

///////////////////////////////////////////////////////////////////////////////
void produce_numbers(hpx::lcos::local::channel_spsc<int>& c2,
    hpx::lcos::local::channel_spsc<int>& c3)
{
    int f1 = 1, f2 = 0;

    int n = channel_get(c2);

    for (int i = 0; i <= n; ++i)
    {
        if (i < 2)
        {
            channel_set(c3, i);
            continue;
        }

        int fib = f1 + f2;
        f2 = f1;
        f1 = fib;

        channel_set(c3, fib);
    }
}

void consume_numbers(int n, hpx::lcos::local::channel_spsc<bool>& c1,
    hpx::lcos::local::channel_spsc<int>& c2,
    hpx::lcos::local::channel_spsc<int>& c3)
{
    channel_set(c2, n);

    for (int i = 0; i <= n; ++i)
    {
        int fib = channel_get(c3);
        HPX_TEST_EQ(fib, verify_fibonacci(i));
    }

    channel_set(c1, true);
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    hpx::lcos::local::channel_spsc<bool> c1(1);
    hpx::lcos::local::channel_spsc<int> c2(1);
    hpx::lcos::local::channel_spsc<int> c3(5);

    hpx::future<void> producer =
        hpx::async(&produce_numbers, std::ref(c2), std::ref(c3));

    hpx::future<void> consumer = hpx::async(
        &consume_numbers, 22, std::ref(c1), std::ref(c2), std::ref(c3));

    hpx::wait_all(producer, consumer);

    HPX_TEST(channel_get(c1));

    return hpx::util::report_errors();
}

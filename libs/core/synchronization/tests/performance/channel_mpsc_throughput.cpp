//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  This work is inspired by https://github.com/aprell/tasking-2.0

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>

#include <hpx/modules/timing.hpp>
#include <hpx/synchronization/channel_mpsc.hpp>

#include <cstddef>
#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
struct data
{
    data() = default;

    explicit data(int d)
    {
        data_[0] = d;
    }

    int data_[8];
};

#if HPX_DEBUG
constexpr int NUM_TESTS = 1000000;
#else
constexpr int NUM_TESTS = 100000000;
#endif

///////////////////////////////////////////////////////////////////////////////
inline data channel_get(hpx::lcos::local::channel_mpsc<data> const& c)
{
    data result;
    while (!c.get(&result))
    {
        hpx::this_thread::yield();
    }
    return result;
}

inline void channel_set(hpx::lcos::local::channel_mpsc<data>& c, data&& val)
{
    while (!c.set(std::move(val)))    // NOLINT
    {
        hpx::this_thread::yield();
    }
}

///////////////////////////////////////////////////////////////////////////////
// Produce
double thread_func_0(hpx::lcos::local::channel_mpsc<data>& c)
{
    std::uint64_t start = hpx::chrono::high_resolution_clock::now();

    for (int i = 0; i != NUM_TESTS; ++i)
    {
        channel_set(c, data{i});
    }

    std::uint64_t end = hpx::chrono::high_resolution_clock::now();

    return static_cast<double>(end - start) / 1e9;
}

// Consume
double thread_func_1(hpx::lcos::local::channel_mpsc<data>& c)
{
    std::uint64_t start = hpx::chrono::high_resolution_clock::now();

    for (int i = 0; i != NUM_TESTS; ++i)
    {
        data d = channel_get(c);
        if (d.data_[0] != i)
        {
            std::cout << "Error!\n";
        }
    }

    std::uint64_t end = hpx::chrono::high_resolution_clock::now();

    return static_cast<double>(end - start) / 1e9;
}

int main()
{
    hpx::lcos::local::channel_mpsc<data> c(10000);

    hpx::future<double> producer = hpx::async(thread_func_0, std::ref(c));
    hpx::future<double> consumer = hpx::async(thread_func_1, std::ref(c));

    auto producer_time = producer.get();
    std::cout << "Producer throughput: " << (NUM_TESTS / producer_time)
              << " [op/s] (" << (producer_time / NUM_TESTS) << " [s/op])\n";

    auto consumer_time = consumer.get();
    std::cout << "Consumer throughput: " << (NUM_TESTS / consumer_time)
              << " [op/s] (" << (consumer_time / NUM_TESTS) << " [s/op])\n";

    return 0;
}

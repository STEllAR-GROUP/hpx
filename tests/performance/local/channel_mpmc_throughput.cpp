//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  This work is inspired by https://github.com/aprell/tasking-2.0

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>

#include <hpx/assertion.hpp>
#include <hpx/lcos/local/channel_mpmc.hpp>
#include <hpx/timing.hpp>

#include <cstddef>
#include <cstdint>

struct data
{
    data() = default;

    data(int d)
    {
        data_[0] = d;
    }

    int data_[8];
};

constexpr int NUM_TESTS = 10000000;

// Produce
double thread_func_0(hpx::lcos::local::channel_mpmc<data>& c)
{
    std::uint64_t start = hpx::util::high_resolution_clock::now();

    for (int i = 0; i != NUM_TESTS; ++i)
    {
        data d{i};
        while (!c.set(std::move(d)))
        {
            hpx::this_thread::yield();
        }
    }

    std::uint64_t end = hpx::util::high_resolution_clock::now();

    return (end - start) / 1e9;
}

// Consume
double thread_func_1(hpx::lcos::local::channel_mpmc<data>& c)
{
    std::uint64_t start = hpx::util::high_resolution_clock::now();

    for (int i = 0; i != NUM_TESTS; ++i)
    {
        data d;
        while (!c.try_get(&d))
        {
            hpx::this_thread::yield();
        }
        HPX_ASSERT(d.data_[0] == i);
    }

    std::uint64_t end = hpx::util::high_resolution_clock::now();

    return (end - start) / 1e9;
}

int main(int argc, char* argv[])
{
    hpx::lcos::local::channel_mpmc<data> c(100);

    hpx::future<double> producer = hpx::async(thread_func_0, std::ref(c));
    hpx::future<double> consumer = hpx::async(thread_func_1, std::ref(c));

    std::cout << "Producer throughput: " << (NUM_TESTS / producer.get())
              << "\n";
    std::cout << "Consumer throughput: " << (NUM_TESTS / consumer.get())
              << "\n";

    return 0;
}

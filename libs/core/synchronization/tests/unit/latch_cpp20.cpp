//  Copyright (c) 2015-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>
#include <hpx/latch.hpp>
#include <hpx/modules/async_local.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <cstddef>
#include <functional>
#include <vector>

#define NUM_THREADS std::size_t(100)

///////////////////////////////////////////////////////////////////////////////
std::atomic<std::size_t> num_threads(0);

///////////////////////////////////////////////////////////////////////////////
void test_arrive_and_wait(hpx::latch& l)
{
    ++num_threads;

    HPX_TEST(!l.try_wait());
    l.arrive_and_wait();
}

void test_count_down(hpx::latch& l)
{
    ++num_threads;

    HPX_TEST(!l.try_wait());
    l.count_down(NUM_THREADS);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    // arraive_and_wait
    {
        hpx::latch l(NUM_THREADS + 1);
        HPX_TEST(!l.try_wait());

        std::vector<hpx::future<void>> results;
        for (std::ptrdiff_t i = 0; i != NUM_THREADS; ++i)
        {
            results.push_back(hpx::async(&test_arrive_and_wait, std::ref(l)));
        }

        HPX_TEST(!l.try_wait());

        // Wait for all threads to reach this point.
        l.arrive_and_wait();

        hpx::wait_all(results);

        HPX_TEST(l.try_wait());
        HPX_TEST_EQ(num_threads.load(), NUM_THREADS);
    }

    // count_down
    {
        num_threads.store(0);

        hpx::latch l(NUM_THREADS + 1);
        HPX_TEST(!l.try_wait());

        hpx::future<void> f = hpx::async(&test_count_down, std::ref(l));

        HPX_TEST(!l.try_wait());
        l.arrive_and_wait();

        f.get();

        HPX_TEST(l.try_wait());
        HPX_TEST_EQ(num_threads.load(), std::size_t(1));
    }

    // wait
    {
        num_threads.store(0);

        hpx::latch l(NUM_THREADS);
        HPX_TEST(!l.try_wait());

        std::vector<hpx::future<void>> results;
        for (std::ptrdiff_t i = 0; i != NUM_THREADS; ++i)
        {
            results.push_back(hpx::async(&test_arrive_and_wait, std::ref(l)));
        }

        hpx::wait_all(results);

        l.wait();

        HPX_TEST(l.try_wait());
        HPX_TEST_EQ(num_threads.load(), NUM_THREADS);
    }

    HPX_TEST_EQ(hpx::local::finalize(), 0);
    return 0;
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

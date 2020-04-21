//  Copyright (c) 2020 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/synchronization/futex.hpp>
#include <hpx/thread_support/atomic_count.hpp>
#include <hpx/threading/thread.hpp>

#include <hpx/testing.hpp>

#include <chrono>
#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

void waiter(hpx::synchronization::futex& f)
{
    f.wait();
}

void timed_waiter(hpx::synchronization::futex& f, hpx::util::steady_duration const& rel_time)
{
    f.wait_for(rel_time);
}


void test_notify_all()
{
    std::size_t num_threads = 10;
    std::vector<hpx::thread> threads;
    threads.reserve(num_threads + 1);

    hpx::util::atomic_count wake_count(0);
    {
        hpx::synchronization::futex f;

        for (std::size_t i = 0; i != num_threads; ++i)
        {
            threads.emplace_back([&f, &wake_count]()
                {
                    waiter(f);
                    ++wake_count;
                });
        }
        threads.emplace_back([&f]()
                {
                    // sleep for 1 second to let the waiter threads actually wait
                    hpx::this_thread::sleep_for(std::chrono::seconds(1));
                    f.notify_all();
                });
        threads.back().join();
    }
    
    for(auto& t: threads)
    {
        if (t.joinable())
            t.join();
    }

    HPX_TEST_EQ(std::size_t(wake_count), num_threads);
}

void test_notify_one()
{
    std::size_t num_threads = 10;
    std::vector<hpx::thread> threads;
    threads.reserve(num_threads + 1);

    hpx::util::atomic_count wake_count(0);
    hpx::util::atomic_count abort_count(0);
    {
        hpx::synchronization::futex f;

        for (std::size_t i = 0; i != num_threads; ++i)
        {
            threads.emplace_back([&f, &wake_count, &abort_count]()
                {
                    try {
                        waiter(f);
                        }
                        catch(...)
                        {
                        ++abort_count;
                        return;
                        }
                    ++wake_count;
                });
        }
        threads.emplace_back([&f]()
                {
                    // sleep for 1 second to let the waiter threads actually wait
                    hpx::this_thread::sleep_for(std::chrono::seconds(1));
                    f.notify_one();
                });
        threads.back().join();
    }
 
    for(auto& t: threads)
    {
        if (t.joinable())
            t.join();
    }

    HPX_TEST_EQ(wake_count, 1);
    HPX_TEST_EQ(std::size_t(abort_count), num_threads-1);
}

void test_timed_wait()
{
    std::size_t num_wakes = 10;
    std::size_t num_threads = 10;
    std::vector<hpx::thread> threads;
    threads.reserve(num_threads);

    hpx::util::atomic_count wake_count(0);
    hpx::synchronization::futex f;

    for (std::size_t i = 0; i != num_threads; ++i)
    {
        threads.emplace_back([&f, &wake_count, num_wakes]()
            {
                for (std::size_t j = 0; j != num_wakes; ++j)
                {
                    timed_waiter(f, std::chrono::milliseconds(10));
                    ++wake_count;
                }
            });
    }

    for(auto& t: threads)
    {
        if (t.joinable())
            t.join();
    }

    HPX_TEST_EQ(std::size_t(wake_count), num_threads * num_wakes);
}

int hpx_main(hpx::program_options::variables_map&)
{
    test_notify_all();
    test_notify_one();
    test_timed_wait();

    hpx::finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Initialize and run HPX
    return hpx::init(argc, argv);
}

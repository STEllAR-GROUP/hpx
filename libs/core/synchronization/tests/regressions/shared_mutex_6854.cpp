//  Copyright (c) 2026 Mo'men Samir
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Regression test for issue #6854.
//
// shared_mutex could hang when shared, upgrade, and exclusive
// locking operations were performed concurrently in tight loops.
//
// This test repeatedly exercises those transitions to ensure progress.

#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/thread_support.hpp>

#include <atomic>
#include <chrono>
#include <mutex>
#include <shared_mutex>
#include <vector>

constexpr unsigned stress_iterations = 1000;
constexpr int reproduction_reps = 1000;

///////////////////////////////////////////////////////////////////////////////
void test_lock_unlock_stress()
{
    hpx::shared_mutex mtx;
    constexpr unsigned num_threads = 4;

    std::atomic<unsigned> completed_count{0};
    std::vector<hpx::future<void>> futures;

    for (unsigned i = 0; i < num_threads; ++i)
    {
        if (i % 2 == 0)
        {
            // Exclusive lock threads
            futures.push_back(hpx::async([&mtx, &completed_count]() {
                for (unsigned j = 0; j < stress_iterations; ++j)
                {
                    std::unique_lock<hpx::shared_mutex> lk(mtx);
                }
                completed_count.fetch_add(1, std::memory_order_relaxed);
            }));
        }
        else
        {
            // Shared lock threads
            futures.push_back(hpx::async([&mtx, &completed_count]() {
                for (unsigned j = 0; j < stress_iterations; ++j)
                {
                    std::shared_lock<hpx::shared_mutex> lk(mtx);
                }
                completed_count.fetch_add(1, std::memory_order_relaxed);
            }));
        }
    }

    hpx::wait_all(futures);
    HPX_TEST_EQ(completed_count.load(), num_threads);
}

///////////////////////////////////////////////////////////////////////////////
void test_lock_unlock_stress_minimal()
{
    for (int r = 0; r < reproduction_reps; ++r)
    {
        hpx::shared_mutex mutex;
        constexpr int n_iter = 3;
        for (int i = 0; i < n_iter; ++i)
        {
            std::vector<hpx::future<void>> tasks;

            tasks.push_back(hpx::async(hpx::launch::async, [&mutex]() {
                std::shared_lock<hpx::shared_mutex> lock(mutex);
            }));

            tasks.push_back(hpx::async(hpx::launch::async, [&mutex]() {
                std::unique_lock<hpx::shared_mutex> lock(mutex);
            }));

            hpx::wait_all(tasks);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_lock_upgrade_stress()
{
    for (int r = 0; r < reproduction_reps; ++r)
    {
        hpx::shared_mutex mutex;
        constexpr int n_iter = 3;
        for (int i = 0; i < n_iter; ++i)
        {
            std::vector<hpx::future<void>> tasks;

            tasks.push_back(hpx::async(hpx::launch::async, [&mutex]() {
                std::shared_lock<hpx::shared_mutex> lock(mutex);
            }));

            tasks.push_back(hpx::async(hpx::launch::async, [&mutex]() {
                mutex.lock_upgrade();
                mutex.unlock_upgrade_and_lock();
                mutex.unlock();
            }));

            hpx::wait_all(tasks);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_lock_unlock_stress();
    test_lock_unlock_stress_minimal();
    test_lock_upgrade_stress();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::local::init(hpx_main, argc, argv), 0);
    return hpx::util::report_errors();
}

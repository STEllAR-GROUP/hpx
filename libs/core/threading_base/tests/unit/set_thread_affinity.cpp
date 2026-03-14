//  Copyright (c) 2024-2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/runtime_local.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/threading_base.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

void test_set_affinity_current_thread()
{
    hpx::async([]() {
        hpx::this_thread::set_affinity(0, hpx::threads::thread_priority::bound);

        auto* data = hpx::threads::get_self_id_data();
        HPX_TEST_EQ(data->get_last_worker_thread_num(), std::size_t(0));
    }).get();
}

void test_set_affinity_other_thread()
{
    hpx::mutex mtx;
    hpx::condition_variable cond;
    bool running = false;

    hpx::thread t([&mtx, &cond, &running]() {
        // signal successful thread initialization
        {
            std::lock_guard<hpx::mutex> lk(mtx);
            running = true;
            cond.notify_all();
        }

        hpx::this_thread::suspend(
            hpx::threads::thread_schedule_state::suspended);

        auto* data = hpx::threads::get_self_id_data();
        HPX_TEST_EQ(data->get_last_worker_thread_num(), 0);
    });

    // wait for the thread to run
    {
        std::unique_lock<hpx::mutex> lk(mtx);
        // NOLINTNEXTLINE(bugprone-infinite-loop)
        while (!running)
            cond.wait(lk);
    }

    hpx::threads::thread_id_type id = t.native_handle();

    hpx::threads::set_thread_affinity(
        id, 0, hpx::threads::thread_priority::bound);

    hpx::threads::set_thread_state(
        id, hpx::threads::thread_schedule_state::pending);

    t.join();
}

void test_set_affinity_all_cores()
{
    hpx::async([]() {
        std::size_t num_pus = hpx::get_os_thread_count();
        for (std::size_t i = 0; i < num_pus; ++i)
        {
            hpx::this_thread::set_affinity(static_cast<std::int16_t>(i),
                hpx::threads::thread_priority::bound);

            auto* data = hpx::threads::get_self_id_data();
            HPX_TEST_EQ(data->get_last_worker_thread_num(), i);
        }
    }).get();
}

void test_concurrent_affinity()
{
    std::vector<hpx::future<void>> futures;
    std::size_t num_threads = 20;

    for (std::size_t i = 0; i < num_threads; ++i)
    {
        futures.push_back(hpx::async([i]() {
            hpx::this_thread::set_affinity(
                static_cast<std::int16_t>(i % hpx::get_os_thread_count()));
        }));
    }
    hpx::wait_all(futures);
    HPX_TEST(true);
}

void test_invalid_pu()
{
    hpx::error_code ec(hpx::throwmode::lightweight);
    hpx::this_thread::set_affinity(
        10000, hpx::threads::thread_priority::bound, ec);
    HPX_TEST(ec);
}

int hpx_main()
{
    test_set_affinity_current_thread();
    test_set_affinity_other_thread();
    test_set_affinity_all_cores();
    test_concurrent_affinity();
    test_invalid_pu();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    hpx::local::init(hpx_main, argc, argv);
    return hpx::util::report_errors();
}

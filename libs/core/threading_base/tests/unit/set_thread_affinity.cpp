//  Copyright (c) 2024-2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/async.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/runtime_local.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/threading_base/set_thread_affinity.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/threading_base/thread_num_tss.hpp>
#include <iostream>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

void test_set_affinity_current_thread()
{
    hpx::async([]() {
        hpx::this_thread::set_affinity(0, hpx::threads::thread_priority::bound);

        auto id = hpx::threads::get_self_id();
        auto* data = hpx::threads::get_thread_id_data(id);
        HPX_TEST_EQ(data->get_last_worker_thread_num(), std::size_t(0));
    }).get();
}

void test_set_affinity_other_thread()
{
    hpx::thread t([]() { hpx::this_thread::suspend(); });

    hpx::threads::thread_id_type id = t.native_handle();

    hpx::threads::set_thread_affinity(id, 0);

    hpx::threads::set_thread_state(
        id, hpx::threads::thread_schedule_state::pending);

    t.join();
    HPX_TEST(true);
}

void test_set_affinity_all_cores()
{
    hpx::async([]() {
        std::size_t num_pus = hpx::threads::hardware_concurrency();
        for (std::size_t i = 0; i < num_pus; ++i)
        {
            hpx::this_thread::set_affinity(static_cast<std::int16_t>(i),
                hpx::threads::thread_priority::bound);
            auto id = hpx::threads::get_self_id();
            auto* data = hpx::threads::get_thread_id_data(id);
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
            hpx::this_thread::set_affinity(static_cast<std::int16_t>(
                i % hpx::threads::hardware_concurrency()));
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
#endif

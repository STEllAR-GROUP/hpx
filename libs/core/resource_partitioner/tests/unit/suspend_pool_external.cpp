//  Copyright (c) 2018 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/chrono.hpp>
#include <hpx/execution.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/schedulers.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/thread_pool_util.hpp>
#include <hpx/modules/threadmanager.hpp>
#include <hpx/thread.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

void test_scheduler(
    int argc, char* argv[], hpx::resource::scheduling_policy scheduler)
{
    hpx::local::init_params init_args;

    init_args.cfg = {"hpx.os_threads=" +
        std::to_string(((std::min)(std::size_t(4),
            std::size_t(hpx::threads::hardware_concurrency()))))};
    init_args.rp_callback = [scheduler](auto& rp,
                                hpx::program_options::variables_map const&) {
        rp.create_thread_pool("default", scheduler);
    };

    hpx::local::start(nullptr, argc, argv, init_args);

    hpx::threads::thread_pool_base& default_pool =
        hpx::resource::get_thread_pool("default");
    std::size_t const default_pool_threads =
        hpx::resource::get_num_threads("default");

    hpx::chrono::high_resolution_timer t;

    while (t.elapsed() < 2)
    {
        for (std::size_t i = 0; i < default_pool_threads * 10000; ++i)
        {
            hpx::post([]() {});
        }

        bool suspended = false;
        hpx::threads::suspend_pool_cb(
            default_pool, [&suspended]() { suspended = true; });

        // NOLINTNEXTLINE(bugprone-infinite-loop)
        while (!suspended)
        {
            std::this_thread::yield();
        }

        bool resumed = false;
        hpx::threads::resume_pool_cb(
            default_pool, [&resumed]() { resumed = true; });

        // NOLINTNEXTLINE(bugprone-infinite-loop)
        while (!resumed)
        {
            std::this_thread::yield();
        }
    }

    hpx::post([]() { hpx::local::finalize(); });

    HPX_TEST_EQ(hpx::local::stop(), 0);
}

int main(int argc, char* argv[])
{
    std::vector<hpx::resource::scheduling_policy> schedulers = {
        hpx::resource::scheduling_policy::local,
        hpx::resource::scheduling_policy::local_priority_fifo,
#if defined(HPX_HAVE_CXX11_STD_ATOMIC_128BIT)
        hpx::resource::scheduling_policy::local_priority_lifo,
#endif
#if defined(HPX_HAVE_CXX11_STD_ATOMIC_128BIT)
        hpx::resource::scheduling_policy::abp_priority_fifo,
        hpx::resource::scheduling_policy::abp_priority_lifo,
#endif
        hpx::resource::scheduling_policy::static_,
        hpx::resource::scheduling_policy::static_priority,
        hpx::resource::scheduling_policy::shared_priority,
    };

    for (auto const scheduler : schedulers)
    {
        test_scheduler(argc, argv, scheduler);
    }

    return hpx::util::report_errors();
}

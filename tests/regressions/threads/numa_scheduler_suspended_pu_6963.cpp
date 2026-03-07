//  Copyright (c) 2026 Vansh Dobhal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Regression test for #6963:
// shared_priority_queue_scheduler: NUMA scheduling does not handle suspended
// processing units. This test suspends PU 0 and then schedules tasks using
// thread_schedule_hint_mode::numa. This ensures the NUMA scheduling path
// correctly falls back through select_active_pu() when a hinted processing
// unit is inactive.

#include <hpx/execution.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/resource_partitioner.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/thread_pool_util.hpp>
#include <hpx/thread.hpp>

#include <atomic>
#include <cstddef>
#include <iostream>
#include <vector>

int hpx_main()
{
    std::size_t const num_threads = hpx::resource::get_num_threads("default");
    if (num_threads < 2)
    {
        return hpx::local::finalize();
    }

    hpx::threads::thread_pool_base& tp =
        hpx::resource::get_thread_pool("default");

    // Deactivate PU 0 so that the NUMA hint path must handle a suspended core.
    hpx::threads::suspend_processing_unit(tp, 0).get();

    std::size_t const num_tasks = 64;
    std::atomic<std::size_t> count{0};
    std::vector<hpx::future<void>> futures;
    futures.reserve(num_tasks);

    for (std::size_t i = 0; i != num_tasks; ++i)
    {
        hpx::execution::parallel_policy_executor<hpx::launch> exec{
            hpx::threads::thread_schedule_hint(
                hpx::threads::thread_schedule_hint_mode::numa,
                static_cast<std::int16_t>(i % 2))};
        futures.push_back(hpx::async(exec, [&count]() { ++count; }));
    }

    hpx::wait_all(futures);
    HPX_TEST_EQ(count, num_tasks);

    // Clean up: reactivate the core before finishing
    hpx::threads::resume_processing_unit(tp, 0).get();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    if (hpx::threads::hardware_concurrency() < 2)
    {
        std::cerr << "[SKIP] Test requires >= 2 hardware threads\n";
        return hpx::util::report_errors();
    }

    // suspend_processing_unit() requires enable_elasticity on the pool.
    hpx::local::init_params iparams;
    iparams.rp_callback = [](auto& rp,
                              hpx::program_options::variables_map const&) {
        rp.create_thread_pool("default",
            hpx::resource::scheduling_policy::shared_priority,
            hpx::threads::policies::scheduler_mode::default_ |
                hpx::threads::policies::scheduler_mode::enable_elasticity);
    };

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, iparams), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
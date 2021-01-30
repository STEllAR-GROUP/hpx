//  Copyright (c) 2017 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threadmanager.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/modules/schedulers.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/timing.hpp>
#include <hpx/threading_base/scheduler_mode.hpp>
#include <hpx/threading_base/thread_helpers.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

int hpx_main()
{
    bool exception_thrown = false;

    try
    {
        // Use .get() to throw exception
        hpx::threads::suspend_pool(*hpx::this_thread::get_pool()).get();
        HPX_TEST_MSG(false, "Suspending should not be allowed on own pool");
    }
    catch (hpx::exception const&)
    {
        exception_thrown = true;
    }

    HPX_TEST(exception_thrown);

    hpx::threads::thread_pool_base& worker_pool =
        hpx::resource::get_thread_pool("worker");
    hpx::execution::parallel_executor worker_exec(
        &hpx::resource::get_thread_pool("worker"));
    std::size_t const worker_pool_threads =
        hpx::resource::get_num_threads("worker");

    {
        // Suspend and resume pool with future
        hpx::chrono::high_resolution_timer t;

        while (t.elapsed() < 1)
        {
            std::vector<hpx::future<void>> fs;

            for (std::size_t i = 0; i < worker_pool_threads * 10000; ++i)
            {
                fs.push_back(hpx::async(worker_exec, []() {}));
            }

            hpx::threads::suspend_pool(worker_pool).get();

            // All work should be done when pool has been suspended
            HPX_TEST(hpx::when_all(std::move(fs)).is_ready());

            hpx::threads::resume_pool(worker_pool).get();
        }
    }

    {
        // Suspend and resume pool with callback
        hpx::lcos::local::counting_semaphore sem;
        hpx::chrono::high_resolution_timer t;

        while (t.elapsed() < 1)
        {
            std::vector<hpx::future<void>> fs;

            for (std::size_t i = 0; i < worker_pool_threads * 10000; ++i)
            {
                fs.push_back(hpx::async(worker_exec, []() {}));
            }

            hpx::threads::suspend_pool_cb(
                worker_pool, [&sem]() { sem.signal(); });

            sem.wait(1);

            // All work should be done when pool has been suspended
            HPX_TEST(hpx::when_all(std::move(fs)).is_ready());

            hpx::threads::resume_pool_cb(
                worker_pool, [&sem]() { sem.signal(); });

            sem.wait(1);
        }
    }

    {
        // Suspend pool with some threads already suspended
        hpx::chrono::high_resolution_timer t;

        while (t.elapsed() < 1)
        {
            for (std::size_t thread_num = 0;
                 thread_num < worker_pool_threads - 1; ++thread_num)
            {
                hpx::threads::suspend_processing_unit(worker_pool, thread_num);
            }

            std::vector<hpx::future<void>> fs;

            for (std::size_t i = 0;
                 i < hpx::resource::get_num_threads("default") * 10000; ++i)
            {
                fs.push_back(hpx::async(worker_exec, []() {}));
            }

            hpx::threads::suspend_pool(worker_pool).get();

            // All work should be done when pool has been suspended
            HPX_TEST(hpx::when_all(std::move(fs)).is_ready());

            hpx::threads::resume_pool(worker_pool).get();
        }
    }

    return hpx::finalize();
}

void test_scheduler(
    int argc, char* argv[], hpx::resource::scheduling_policy scheduler)
{
    hpx::init_params init_args;

    init_args.cfg = {"hpx.os_threads=4"};
    init_args.rp_callback = [scheduler](auto& rp,
                                hpx::program_options::variables_map const&) {
        rp.create_thread_pool("worker", scheduler);

        int const worker_pool_threads = 3;
        int worker_pool_threads_added = 0;

        for (hpx::resource::numa_domain const& d : rp.numa_domains())
        {
            for (hpx::resource::core const& c : d.cores())
            {
                for (hpx::resource::pu const& p : c.pus())
                {
                    if (worker_pool_threads_added < worker_pool_threads)
                    {
                        rp.add_resource(p, "worker");
                        ++worker_pool_threads_added;
                    }
                }
            }
        }
    };

    HPX_TEST_EQ(hpx::init(argc, argv, init_args), 0);
}

int main(int argc, char* argv[])
{
    std::vector<hpx::resource::scheduling_policy> schedulers = {
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
        hpx::resource::scheduling_policy::local,
        hpx::resource::scheduling_policy::local_priority_fifo,
#if defined(HPX_HAVE_CXX11_STD_ATOMIC_128BIT)
        hpx::resource::scheduling_policy::local_priority_lifo,
#endif
#endif
#if defined(HPX_HAVE_ABP_SCHEDULER) && defined(HPX_HAVE_CXX11_STD_ATOMIC_128BIT)
        hpx::resource::scheduling_policy::abp_priority_fifo,
        hpx::resource::scheduling_policy::abp_priority_lifo,
#endif
#if defined(HPX_HAVE_STATIC_SCHEDULER)
        hpx::resource::scheduling_policy::static_,
#endif
#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
        hpx::resource::scheduling_policy::static_priority,
#endif
#if defined(HPX_HAVE_SHARED_PRIORITY_SCHEDULER)
        hpx::resource::scheduling_policy::shared_priority,
#endif
    };

    for (auto const scheduler : schedulers)
    {
        test_scheduler(argc, argv, scheduler);
    }

    return hpx::util::report_errors();
}

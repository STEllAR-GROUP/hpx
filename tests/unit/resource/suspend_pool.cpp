//  Copyright (c) 2017 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/compat/thread.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threadmanager.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/policies/schedulers.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

int hpx_main(int argc, char* argv[])
{
    bool exception_thrown = false;

    try
    {
        hpx::this_thread::get_pool()->suspend();
        HPX_TEST_MSG(false, "Suspending should not be allowed on own pool");
    }
    catch (hpx::exception const&)
    {
        exception_thrown = true;
    }

    HPX_TEST(exception_thrown);

    hpx::threads::detail::thread_pool_base& worker_pool =
        hpx::resource::get_thread_pool("worker");
    hpx::threads::executors::pool_executor worker_exec("worker");
    std::size_t const worker_pool_threads =
        hpx::resource::get_num_threads("worker");

    {
        // Suspend and resume pool with future
        hpx::util::high_resolution_timer t;

        while (t.elapsed() < 5)
        {
            std::vector<hpx::future<void>> fs;

            for (std::size_t i = 0;
                 i < worker_pool_threads * 10000; ++i)
            {
                fs.push_back(hpx::async(worker_exec, [](){}));
            }

            worker_pool.suspend().wait();

            // All work should be done when pool has been suspended
            HPX_TEST(hpx::when_all(std::move(fs)).is_ready());

            worker_pool.resume();
        }
    }

    {
        // Suspend and resume pool with callback
        hpx::lcos::local::counting_semaphore sem;
        hpx::util::high_resolution_timer t;

        while (t.elapsed() < 5)
        {
            std::vector<hpx::future<void>> fs;

            for (std::size_t i = 0;
                 i < worker_pool_threads * 10000; ++i)
            {
                fs.push_back(hpx::async(worker_exec, [](){}));
            }

            worker_pool.suspend_cb([&sem]()
                {
                    sem.signal();
                });

            sem.wait(1);

            // All work should be done when pool has been suspended
            HPX_TEST(hpx::when_all(std::move(fs)).is_ready());

            worker_pool.resume();
        }
    }

    {
        // Suspend pool with some threads already suspended
        hpx::util::high_resolution_timer t;

        while (t.elapsed() < 5)
        {
            for (std::size_t thread_num = 0;
                thread_num < worker_pool_threads - 1; ++thread_num)
            {
                worker_pool.suspend_processing_unit(thread_num);
            }

            std::vector<hpx::future<void>> fs;

            for (std::size_t i = 0;
                 i < hpx::resource::get_num_threads("default") * 10000; ++i)
            {
                fs.push_back(hpx::async(worker_exec, [](){}));
            }

            worker_pool.suspend().wait();

            // All work should be done when pool has been suspended
            HPX_TEST(hpx::when_all(std::move(fs)).is_ready());

            worker_pool.resume();
        }
    }

    return hpx::finalize();
}

template <typename Scheduler>
void test_scheduler(int argc, char* argv[])
{
    std::vector<std::string> cfg =
    {
        "hpx.os_threads=4"
    };

    hpx::resource::partitioner rp(argc, argv, std::move(cfg));

    rp.create_thread_pool("worker",
        [](hpx::threads::policies::callback_notifier& notifier,
            std::size_t num_threads, std::size_t thread_offset,
            std::size_t pool_index, std::string const& pool_name)
        -> std::unique_ptr<hpx::threads::detail::thread_pool_base>
        {
            typename Scheduler::init_parameter_type init(num_threads);
            std::unique_ptr<Scheduler> scheduler(new Scheduler(init));

            auto mode = hpx::threads::policies::scheduler_mode(
                hpx::threads::policies::do_background_work |
                hpx::threads::policies::reduce_thread_priority |
                hpx::threads::policies::delay_exit |
                hpx::threads::policies::enable_elasticity);

            std::unique_ptr<hpx::threads::detail::thread_pool_base> pool(
                new hpx::threads::detail::scheduled_thread_pool<Scheduler>(
                    std::move(scheduler), notifier, pool_index, pool_name, mode,
                    thread_offset));

            return pool;
        });

    int const worker_pool_threads = 3;
    int worker_pool_threads_added = 0;

    for (const hpx::resource::numa_domain& d : rp.numa_domains())
    {
        for (const hpx::resource::core& c : d.cores())
        {
            for (const hpx::resource::pu& p : c.pus())
            {
                if (worker_pool_threads_added < worker_pool_threads)
                {
                    rp.add_resource(p, "worker");
                    ++worker_pool_threads_added;
                }
            }
        }
    }

    HPX_TEST_EQ(hpx::init(argc, argv), 0);
}

int main(int argc, char* argv[])
{
    test_scheduler<hpx::threads::policies::local_queue_scheduler<>>(argc, argv);
    test_scheduler<hpx::threads::policies::local_priority_queue_scheduler<>>(argc,
        argv);

    return hpx::util::report_errors();
}

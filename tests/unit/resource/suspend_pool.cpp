//  Copyright (c) 2017 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/hpx_start.hpp>
#include <hpx/include/async.hpp>
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
#include <string>
#include <utility>
#include <vector>

int hpx_main(int argc, char* argv[])
{
    try
    {
        hpx::this_thread::get_pool()->suspend();
        HPX_TEST_MSG(false, "Suspending is not possible on own pool");
    }
    catch (std::runtime_error const&)
    {
    }

    hpx::threads::detail::thread_pool_base& worker_pool =
        hpx::resource::get_thread_pool("worker");
    hpx::threads::executors::pool_executor worker_exec("worker");

    hpx::util::high_resolution_timer t;

    while (t.elapsed() < 5)
    {
        std::vector<hpx::future<void>> fs;

        for (std::size_t i = 0;
            i < hpx::resource::get_num_threads("default") * 100000; ++i)
        {
            fs.push_back(hpx::async(worker_exec, [](){}));
        }

        worker_pool.suspend();

        // Can only suspend once all work is done
        auto f = hpx::when_all(std::move(fs));
        HPX_TEST(f.is_ready());

        worker_pool.resume();
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg =
    {
        "hpx.os_threads=4"
    };

    hpx::resource::partitioner rp(argc, argv, std::move(cfg));

    rp.create_thread_pool("default", hpx::resource::scheduling_policy::local_priority_lifo);
    rp.create_thread_pool("worker", hpx::resource::scheduling_policy::local_priority_lifo);

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

    return hpx::util::report_errors();
}

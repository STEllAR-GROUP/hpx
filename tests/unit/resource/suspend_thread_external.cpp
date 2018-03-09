//  Copyright (c) 2017 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/policies/schedulers.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

int hpx_main(int argc, char* argv[])
{
    std::size_t const num_threads = hpx::resource::get_num_threads("worker");

    HPX_TEST_EQ(std::size_t(3), num_threads);

    hpx::threads::thread_pool_base& tp =
        hpx::resource::get_thread_pool("worker");

    {
        // Check number of used resources
        for (std::size_t thread_num = 0; thread_num < num_threads - 1; ++thread_num)
        {
            tp.suspend_processing_unit(thread_num).get();
            HPX_TEST_EQ(std::size_t(num_threads - thread_num - 1),
                tp.get_active_os_thread_count());
        }

        for (std::size_t thread_num = 0; thread_num < num_threads - 1; ++thread_num)
        {
            tp.resume_processing_unit(thread_num).get();
            HPX_TEST_EQ(std::size_t(thread_num + 2),
                tp.get_active_os_thread_count());
        }
    }

    {
        // Check suspending and resuming the same thread without waiting for
        // each to finish.
        for (std::size_t thread_num = 0;
             thread_num < hpx::resource::get_num_threads("worker");
             ++thread_num)
        {
            std::vector<hpx::future<void>> fs;

            fs.push_back(tp.suspend_processing_unit(thread_num));
            fs.push_back(tp.resume_processing_unit(thread_num));

            hpx::wait_all(fs);

            // Suspend is not guaranteed to run before resume, so make sure
            // processing unit is running
            tp.resume_processing_unit(thread_num).get();

            fs.clear();

            // Launching 4 (i.e. same as number of threads) tasks may deadlock
            // as no thread is available to steal from the current thread.
            fs.push_back(tp.suspend_processing_unit(thread_num));
            fs.push_back(tp.suspend_processing_unit(thread_num));
            fs.push_back(tp.suspend_processing_unit(thread_num));

            hpx::wait_all(fs);

            fs.clear();

            // Launching 4 (i.e. same as number of threads) tasks may deadlock
            // as no thread is available to steal from the current thread.
            fs.push_back(tp.resume_processing_unit(thread_num));
            fs.push_back(tp.resume_processing_unit(thread_num));
            fs.push_back(tp.resume_processing_unit(thread_num));

            hpx::wait_all(fs);
        }
    }

    {
        // Check random scheduling with reducing resources.
        std::size_t thread_num = 0;
        bool up = true;
        std::vector<hpx::future<void>> fs;
        hpx::util::high_resolution_timer t;
        while (t.elapsed() < 2)
        {
            for (std::size_t i = 0;
                i < hpx::resource::get_num_threads("worker") * 10;
                ++i)
            {
                fs.push_back(hpx::async([](){}));
            }

            if (up)
            {
                if (thread_num < hpx::resource::get_num_threads("worker"))
                {
                    tp.suspend_processing_unit(thread_num).get();
                }

                ++thread_num;

                if (thread_num == hpx::resource::get_num_threads("worker"))
                {
                    up = false;
                    --thread_num;
                }
            }
            else
            {
                tp.resume_processing_unit(thread_num - 1).get();

                --thread_num;

                if (thread_num == 0)
                {
                    up = true;
                }
            }
        }

        hpx::when_all(std::move(fs)).get();

        // Don't exit with suspended pus
        for (std::size_t thread_num_resume = 0; thread_num_resume < thread_num;
            ++thread_num_resume)
        {
            tp.resume_processing_unit(thread_num_resume).get();
        }
    }

    return hpx::finalize();
}

void test_scheduler(int argc, char* argv[],
    hpx::resource::scheduling_policy scheduler)
{
    std::vector<std::string> cfg =
    {
        "hpx.os_threads=4"
    };

    hpx::resource::partitioner rp(argc, argv, std::move(cfg));

    rp.create_thread_pool("worker", scheduler,
        hpx::threads::policies::scheduler_mode(
            hpx::threads::policies::default_mode |
            hpx::threads::policies::enable_elasticity));

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
    std::vector<hpx::resource::scheduling_policy> schedulers =
        {
            hpx::resource::scheduling_policy::local,
            hpx::resource::scheduling_policy::local_priority_fifo,
            hpx::resource::scheduling_policy::local_priority_lifo,
            hpx::resource::scheduling_policy::abp_priority_fifo,
            hpx::resource::scheduling_policy::abp_priority_lifo,
            hpx::resource::scheduling_policy::static_,
            hpx::resource::scheduling_policy::static_priority,
        };

    for (auto const scheduler : schedulers)
    {
        test_scheduler(argc, argv, scheduler);
    }

    return hpx::util::report_errors();
}

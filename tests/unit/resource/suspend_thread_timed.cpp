//  Copyright (c) 2017 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/modules/schedulers.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/timing.hpp>
#include <hpx/threading_base/scheduler_mode.hpp>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

int hpx_main(int argc, char* argv[])
{
    hpx::threads::thread_pool_base& worker_pool =
        hpx::resource::get_thread_pool("default");
    std::cout << "Starting test with scheduler "
              << worker_pool.get_scheduler()->get_description() << std::endl;
    std::size_t const num_threads = hpx::resource::get_num_threads("default");

    HPX_TEST_EQ(std::size_t(4), num_threads);

    hpx::threads::thread_pool_base& tp =
        hpx::resource::get_thread_pool("default");

    {
        // Check random scheduling with reducing resources.
        std::size_t thread_num = 0;
        bool up = true;
        std::vector<hpx::future<void>> fs;

        hpx::execution::parallel_executor exec(
            hpx::resource::get_thread_pool("default"));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(1, 100);

        hpx::chrono::high_resolution_timer t;

        while (t.elapsed() < 1)
        {
            for (std::size_t i = 0;
                 i < hpx::resource::get_num_threads("default"); ++i)
            {
                fs.push_back(hpx::parallel::execution::async_execute_after(
                    exec, std::chrono::milliseconds(dist(gen)), []() {}));
            }

            if (up)
            {
                if (thread_num != hpx::resource::get_num_threads("default") - 1)
                {
                    hpx::threads::suspend_processing_unit(tp, thread_num).get();
                }

                ++thread_num;

                if (thread_num == hpx::resource::get_num_threads("default"))
                {
                    up = false;
                    --thread_num;
                }
            }
            else
            {
                hpx::threads::resume_processing_unit(tp, thread_num - 1).get();

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
            hpx::threads::resume_processing_unit(tp, thread_num_resume).get();
        }
    }

    return hpx::finalize();
}

void test_scheduler(
    int argc, char* argv[], hpx::resource::scheduling_policy scheduler)
{
    hpx::init_params init_args;

    init_args.cfg = {"hpx.os_threads=4"};
    init_args.rp_callback = [scheduler](auto& rp) {
        std::cout << "\nCreating pool with scheduler " << scheduler
                  << std::endl;

        rp.create_thread_pool("default", scheduler,
            hpx::threads::policies::scheduler_mode(
                hpx::threads::policies::default_mode |
                hpx::threads::policies::enable_elasticity));
    };

    HPX_TEST_EQ(hpx::init(argc, argv, init_args), 0);
}

int main(int argc, char* argv[])
{
    // NOTE: Static schedulers do not support suspending the own worker thread
    // because they do not steal work. Periodic priority scheduler not tested
    // because it does not take into account scheduler states when scheduling
    // work.

    {
        // These schedulers should succeed
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
        };

        for (auto const scheduler : schedulers)
        {
            test_scheduler(argc, argv, scheduler);
        }
    }

    {
        // These schedulers should fail
        std::vector<hpx::resource::scheduling_policy> schedulers = {
#if defined(HPX_HAVE_STATIC_SCHEDULER)
            hpx::resource::scheduling_policy::static_,
#endif
#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
            hpx::resource::scheduling_policy::static_priority,
#endif
#if defined(HPX_HAVE_SHARED_PRIORITY_SCHEDULER)
        // until timed thread problems are fix, disable this
        //hpx::resource::scheduling_policy::shared_priority,
#endif
        };

        for (auto const scheduler : schedulers)
        {
            bool exception_thrown = false;
            try
            {
                test_scheduler(argc, argv, scheduler);
            }
            catch (hpx::exception const&)
            {
                exception_thrown = true;
            }

            HPX_TEST(exception_thrown);
        }
    }

    return hpx::util::report_errors();
}

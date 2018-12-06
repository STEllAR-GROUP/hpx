//  Copyright (c) 2017 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/policies/schedulers.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <chrono>
#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

int hpx_main(int argc, char* argv[])
{
    std::size_t const num_threads = hpx::resource::get_num_threads("default");

    HPX_TEST_EQ(std::size_t(4), num_threads);

    hpx::threads::thread_pool_base& tp =
        hpx::resource::get_thread_pool("default");

    {
        // Check random scheduling with reducing resources.
        std::size_t thread_num = 0;
        bool up = true;
        std::vector<hpx::future<void>> fs;

        hpx::threads::executors::pool_executor exec("default");

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(1,100);

        hpx::util::high_resolution_timer t;

        while (t.elapsed() < 1)
        {
            for (std::size_t i = 0;
                i < hpx::resource::get_num_threads("default"); ++i)
            {
                fs.push_back(hpx::parallel::execution::async_execute_after(
                    exec, std::chrono::milliseconds(dist(gen)), [](){}));
            }

            if (up)
            {
                if (thread_num != hpx::resource::get_num_threads("default") - 1)
                {
                    tp.suspend_processing_unit(thread_num).get();
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

    rp.create_thread_pool("default", scheduler,
         hpx::threads::policies::scheduler_mode(
             hpx::threads::policies::default_mode |
             hpx::threads::policies::enable_elasticity));

    HPX_TEST_EQ(hpx::init(argc, argv), 0);
}

int main(int argc, char* argv[])
{
    // NOTE: Static schedulers do not support suspending the own worker thread
    // because they do not steal work. Periodic priority scheduler not tested
    // because it does not take into account scheduler states when scheduling
    // work.

    {
        // These schedulers should succeed
        std::vector<hpx::resource::scheduling_policy> schedulers =
            {
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
                hpx::resource::scheduling_policy::local,
                hpx::resource::scheduling_policy::local_priority_fifo,
                hpx::resource::scheduling_policy::local_priority_lifo,
#endif
#if defined(HPX_HAVE_ABP_SCHEDULER)
                hpx::resource::scheduling_policy::abp_priority_fifo,
                hpx::resource::scheduling_policy::abp_priority_lifo,
#endif
#if defined(HPX_HAVE_SHARED_PRIORITY_SCHEDULER)
                hpx::resource::scheduling_policy::shared_priority,
#endif
            };

        for (auto const scheduler : schedulers)
        {
            test_scheduler(argc, argv, scheduler);
        }
    }

    {
        // These schedulers should fail
        std::vector<hpx::resource::scheduling_policy> schedulers =
        {
#if defined(HPX_HAVE_STATIC_SCHEDULER)
            hpx::resource::scheduling_policy::static_,
#endif
#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
            hpx::resource::scheduling_policy::static_priority,
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

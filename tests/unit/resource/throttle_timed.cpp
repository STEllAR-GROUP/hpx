//  Copyright (c) 2017 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/lcos/when_all.hpp>
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

template <typename Scheduler>
void test_scheduler(int argc, char* argv[])
{
    std::vector<std::string> cfg =
    {
        "hpx.os_threads=4"
    };

    hpx::resource::partitioner rp(argc, argv, std::move(cfg));

    rp.create_thread_pool("default",
        [](hpx::threads::policies::callback_notifier& notifier,
            std::size_t num_threads, std::size_t thread_offset,
            std::size_t pool_index, std::string const& pool_name)
        -> std::unique_ptr<hpx::threads::thread_pool_base>
        {
            typename Scheduler::init_parameter_type init(num_threads);
            std::unique_ptr<Scheduler> scheduler(new Scheduler(init));

            auto mode = hpx::threads::policies::scheduler_mode(
                hpx::threads::policies::do_background_work |
                hpx::threads::policies::reduce_thread_priority |
                hpx::threads::policies::delay_exit |
                hpx::threads::policies::enable_elasticity);

            std::unique_ptr<hpx::threads::thread_pool_base> pool(
                new hpx::threads::detail::scheduled_thread_pool<Scheduler>(
                    std::move(scheduler), notifier, pool_index, pool_name, mode,
                    thread_offset));

            return pool;
        });

    HPX_TEST_EQ(hpx::init(argc, argv), 0);
}

int main(int argc, char* argv[])
{
    // NOTE: Static schedulers do not support suspending the own worker thread
    // because they do not steal work. Periodic priority scheduler not tested
    // because it does not take into account scheduler states when scheduling
    // work.

    test_scheduler<hpx::threads::policies::local_queue_scheduler<>>(argc, argv);
    test_scheduler<hpx::threads::policies::local_priority_queue_scheduler<>>(argc,
        argv);

    {
        bool exception_thrown = false;
        try
        {
            test_scheduler<hpx::threads::policies::static_queue_scheduler<>>(argc,
                argv);
        }
        catch (hpx::exception const&)
        {
            exception_thrown = true;
        }

        HPX_TEST(exception_thrown);
    }

    {
        bool exception_thrown = false;
        try
        {
            test_scheduler<
                hpx::threads::policies::static_priority_queue_scheduler<>
            >(argc, argv);
        }
        catch (hpx::exception const&)
        {
            exception_thrown = true;
        }

        HPX_TEST(exception_thrown);
    }

    return hpx::util::report_errors();
}

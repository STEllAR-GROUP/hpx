//  Copyright (c) 2017 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <chrono>
#include <cstddef>
#include <random>
#include <string>
#include <utility>
#include <vector>

int hpx_main(int argc, char* argv[])
{
    std::size_t const num_threads = hpx::resource::get_num_threads("default");

    HPX_TEST_EQ(std::size_t(4), num_threads);

    hpx::threads::detail::thread_pool_base& tp =
        hpx::resource::get_thread_pool("default");

    // Enable elasticity
    tp.set_scheduler_mode(
        hpx::threads::policies::scheduler_mode(
            hpx::threads::policies::do_background_work |
            hpx::threads::policies::reduce_thread_priority |
            hpx::threads::policies::delay_exit |
            hpx::threads::policies::enable_elasticity));

    {
        // Check random scheduling with reducing resources.
        std::size_t thread_num = 0;
        bool up = true;
        std::vector<hpx::future<void>> fs;

        hpx::threads::executors::pool_executor exec("default");

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(1,1000);

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
                    tp.suspend_processing_unit(thread_num);
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
                tp.resume_processing_unit(thread_num - 1);

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
            tp.resume_processing_unit(thread_num_resume);
        }
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg =
    {
        "hpx.os_threads=4"
    };

    using hpx::resource::scheduling_policy;

    // TODO: Loop through integer values?
    // TODO: Should all schedulers support suspension?
    std::vector<scheduling_policy> const policies =
    {
        //scheduling_policy::local,
        scheduling_policy::local_priority_fifo,
        scheduling_policy::local_priority_lifo,
        //scheduling_policy::static_,
        //scheduling_policy::static_priority,
        //scheduling_policy::abp_priority,
        //scheduling_policy::hierarchy,
        //scheduling_policy::periodic_priority,
        //scheduling_policy::throttle
    };

    for (auto policy : policies)
    {
        // Set up the resource partitioner
        hpx::resource::partitioner rp(argc, argv, std::move(cfg));
        rp.create_thread_pool("default", policy);

        HPX_TEST_EQ(hpx::init(argc, argv), 0);
    }

    return hpx::util::report_errors();
}

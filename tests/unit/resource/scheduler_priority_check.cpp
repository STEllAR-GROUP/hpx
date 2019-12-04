//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Test that creates a set of tasks using normal priority, but every
// Nth normal task spawns a set of high priority tasks.
// The test is intended to be used with a task plotting/profiling
// tool to verify that high priority tasks run before low ones.

#include <hpx/hpx_init.hpp>

#include <hpx/async.hpp>
#include <hpx/include/parallel_execution.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/program_options.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>
#include <hpx/testing.hpp>
#include <hpx/util/annotated_function.hpp>

#include <atomic>
#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using hpx::program_options::options_description;
using hpx::program_options::value;
using hpx::program_options::variables_map;

// dummy function we will call using async
void dummy_task(const std::string& name, std::size_t n)
{
    hpx::util::annotate_function t(name);
    // no other work can take place on this thread whilst it sleeps
    std::this_thread::sleep_for(std::chrono::microseconds(n));
    //
    for (std::size_t i(0); i < n; ++i)
    {
    }
}

inline std::size_t st_rand()
{
    return std::size_t(std::rand());
}

int hpx_main(variables_map& vm)
{
    auto const sched = hpx::threads::get_self_id_data()->get_scheduler_base();
    std::cout << "Scheduler is " << sched->get_description() << std::endl;
    if (std::string("core-shared_priority_queue_scheduler") ==
        sched->get_description())
    {
        std::cout << "Setting shared-priority mode flags" << std::endl;
        sched->add_remove_scheduler_mode(
            // add these flags
            hpx::threads::policies::scheduler_mode(
                hpx::threads::policies::enable_stealing |
                hpx::threads::policies::enable_stealing_numa |
                hpx::threads::policies::assign_work_round_robin |
                hpx::threads::policies::steal_high_priority_first),
            // remove these flags
            hpx::threads::policies::scheduler_mode(
                hpx::threads::policies::assign_work_thread_parent |
                hpx::threads::policies::steal_after_local |
                hpx::threads::policies::do_background_work |
                hpx::threads::policies::reduce_thread_priority |
                hpx::threads::policies::delay_exit |
                hpx::threads::policies::fast_idle_mode |
                hpx::threads::policies::enable_elasticity));
    }

    // setup executors for different task priorities on the pools
    hpx::threads::scheduled_executor HP_executor =
        hpx::threads::executors::pool_executor(
            "default", hpx::threads::thread_priority_high);

    hpx::threads::scheduled_executor NP_executor =
        hpx::threads::executors::pool_executor(
            "default", hpx::threads::thread_priority_default);

    // randomly create normal priority tasks
    // and then a set of HP tasks in periodic bursts
    // Use task plotting tools to validate that scheduling is correct
    const int np_loop = vm["nnp"].as<int>();
    const int hp_loop = vm["nhp"].as<int>();
    const int np_m = vm["mnp"].as<int>();
    const int hp_m = vm["mhp"].as<int>();
    const int cycles = vm["cycles"].as<int>();

    const int np_total = np_loop * cycles;
    //
    std::atomic<int> counter((np_loop + hp_loop) * cycles);
    std::atomic<int> counter1(0);
    {
        hpx::util::annotate_function annotate("launch");
        for (int i = 0; i < np_total; ++i)
        {
            // normal priority
            auto f3 =
                hpx::async(NP_executor, &dummy_task, "NP task", np_m)
                    .then(hpx::launch::sync, [&](auto&&) {
                        counter--;
                        if (++counter1 % np_loop == 0)
                        {
                            hpx::util::annotate_function annotate("launch");
                            for (int j = 0; j < hp_loop; ++j)
                            {
                                // high priority
                                auto f1 = hpx::async(
                                    HP_executor, &dummy_task, "HP task", hp_m)
                                              .then(hpx::launch::sync,
                                                  [&](auto&&) { counter--; });
                            }
                        }
                    });
        }
    }
    do
    {
        hpx::this_thread::yield();
    } while (counter > 0);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()
        ("nnp", value<int>()->default_value(100),
         "number of Normal Priority futures per cycle")

        ("nhp", value<int>()->default_value(50),
         "number of High Priority futures per cycle")

        ("mnp", value<int>()->default_value(1000),
         "microseconds per Normal Priority task")

        ("mhp", value<int>()->default_value(100),
         "microseconds per High Priority task")

        ("cycles", value<int>()->default_value(10),
         "number of cycles in the benchmark");
    // clang-format on

    // Create the resource partitioner
    hpx::resource::partitioner rp(cmdline, argc, argv);

    HPX_TEST_EQ(hpx::init(), 0);

    return hpx::util::report_errors();
}

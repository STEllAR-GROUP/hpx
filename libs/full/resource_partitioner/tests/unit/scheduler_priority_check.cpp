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

#include <hpx/async_combinators/when_all.hpp>
#include <hpx/include/parallel_execution.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/modules/async_local.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/threading_base/annotated_function.hpp>

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
void dummy_task(std::size_t n)
{
    // no other work can take place on this thread whilst it sleeps
    bool sleep = true;
    auto start = std::chrono::steady_clock::now();
    do
    {
        std::this_thread::sleep_for(std::chrono::microseconds(n) / 25);
        auto now = std::chrono::steady_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(now - start);
        sleep = (elapsed < std::chrono::microseconds(n));
    } while (sleep);
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
    hpx::execution::parallel_executor HP_executor(
        &hpx::resource::get_thread_pool("default"),
        hpx::threads::thread_priority::high);

    hpx::execution::parallel_executor NP_executor(
        &hpx::resource::get_thread_pool("default"),
        hpx::threads::thread_priority::default_);

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
    struct dec_counter
    {
        explicit dec_counter(std::atomic<int>& counter)
          : counter_(counter)
        {
        }
        ~dec_counter()
        {
            --counter_;
        }
        //
        std::atomic<int>& counter_;
    };

    // diagnostic counters for debugging profiler numbers
    std::atomic<int> np_task_count(0);
    std::atomic<int> hp_task_count(0);
    std::atomic<int> hp_launch_count(0);
    std::atomic<int> launch_count(0);
    //
    std::atomic<int> count_down((np_loop + hp_loop) * cycles);
    std::atomic<int> counter(0);
    auto f3 = hpx::async(NP_executor,
        hpx::util::annotated_function(
            [&]() {
                ++launch_count;
                for (int i = 0; i < np_total; ++i)
                {
                    // normal priority
                    auto f3 = hpx::async(NP_executor,
                        hpx::util::annotated_function(
                            [&, np_m]() {
                                np_task_count++;
                                dec_counter dec(count_down);
                                dummy_task(std::size_t(np_m));
                            },
                            "NP task"));

                    // continuation runs as a sync task
                    f3.then(hpx::launch::sync, [&](hpx::future<void>&&) {
                        // on every Nth task, spawn new HP tasks, otherwise quit
                        if ((++counter) % np_loop != 0)
                            return;

                        // Launch HP tasks using an HP task to do it
                        hpx::async(HP_executor,
                            hpx::util::annotated_function(
                                [&]() {
                                    ++hp_launch_count;
                                    for (int j = 0; j < hp_loop; ++j)
                                    {
                                        hpx::async(HP_executor,
                                            hpx::util::annotated_function(
                                                [&]() {
                                                    ++hp_task_count;
                                                    dec_counter dec(count_down);
                                                    dummy_task(
                                                        std::size_t(hp_m));
                                                },
                                                "HP task"));
                                    }
                                },
                                "Launch HP"));
                    });
                }
            },
            "Launch"));

    // wait for everything to finish
    do
    {
        hpx::this_thread::yield();
    } while (count_down > 0);

    std::cout << "Tasks NP  : " << np_task_count << "\n"
              << "Tasks HP  : " << hp_task_count << "\n"
              << "Launch    : " << launch_count << "\n"
              << "Launch HP : " << hp_launch_count << std::endl;

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

    // Setup the init parameters
    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;

    HPX_TEST_EQ(hpx::init(argc, argv, init_args), 0);

    return hpx::util::report_errors();
}

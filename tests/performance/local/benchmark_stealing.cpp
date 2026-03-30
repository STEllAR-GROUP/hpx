//  Copyright (c) 2026 Arpit Khandelwal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/chrono.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <iostream>
#include <vector>

void task_func(std::uint64_t delay_ns)
{
    if (delay_ns == 0)
        return;
    auto start = hpx::chrono::high_resolution_clock::now();
    while ((hpx::chrono::high_resolution_clock::now() - start) < delay_ns)
        ;
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    std::uint64_t tasks = vm["tasks"].as<std::uint64_t>();
    std::uint64_t delay = vm["delay"].as<std::uint64_t>();

    // warm up
    hpx::chrono::high_resolution_timer t;

    std::vector<hpx::future<void>> futures;
    futures.reserve(tasks);

    // Create an executor that schedules on thread 0 to force stealing
    // from other threads
    hpx::execution::parallel_executor exec(
        hpx::threads::thread_schedule_hint(0));

    for (std::uint64_t i = 0; i < tasks; ++i)
    {
        futures.push_back(hpx::async(exec, &task_func, delay));
    }

    hpx::wait_all(futures);
    double elapsed = t.elapsed();

    std::cout << "Tasks: " << tasks << ", Delay: " << delay << "\n";
    std::cout << "Time: " << elapsed << " s\n";
    std::cout << "Throughput: " << tasks / elapsed << " tasks/s\n";

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    hpx::program_options::options_description desc("Usage:");
    desc.add_options()("tasks",
        hpx::program_options::value<std::uint64_t>()->default_value(1000000),
        "Number of tasks")("delay",
        hpx::program_options::value<std::uint64_t>()->default_value(0),
        "Delay in ns");

    hpx::init_params init_args;
    init_args.desc_cmdline = desc;

    return hpx::init(argc, argv, init_args);
}

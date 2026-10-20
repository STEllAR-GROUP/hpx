//  Copyright (c) 2026 Rohan Pattanayak
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #7789: all tasks below
// are placed on worker 0 and run_as_child, so hpx_main can run them directly
// while it waits for them. Worker 0 doesn't get back to its scheduling loop in
// the meantime, so it used to leave the steal requests of the other workers
// unanswered and none of the tasks was stolen.

#include <hpx/chrono.hpp>
#include <hpx/execution.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

constexpr std::size_t num_tasks = 2000;
constexpr std::uint64_t task_duration_ns = 20000;

std::atomic<std::size_t> num_stolen(0);

void busy_task()
{
    auto const start = hpx::chrono::high_resolution_clock::now();
    while (hpx::chrono::high_resolution_clock::now() - start < task_duration_ns)
    {
    }

    if (hpx::get_worker_thread_num() != 0)
    {
        ++num_stolen;
    }
}

int hpx_main()
{
    hpx::threads::thread_schedule_hint hint(0);
    hint.runs_as_child_mode(hpx::threads::thread_execution_hint::run_as_child);
    hpx::execution::parallel_executor exec(hint);

    std::vector<hpx::future<void>> futures;
    futures.reserve(num_tasks);
    for (std::size_t i = 0; i != num_tasks; ++i)
    {
        futures.push_back(hpx::async(exec, &busy_task));
    }
    hpx::wait_all(futures);

    // a lower bound only, to stay robust on slow or oversubscribed machines
    HPX_TEST(num_stolen > 0);

    return hpx::local::finalize();
}

int main(int, char* argv[])
{
    for (char const* scheduler :
        {"local-workrequesting-fifo", "local-workrequesting-mc"})
    {
        num_stolen = 0;

        std::string const queuing = std::string("--hpx:queuing=") + scheduler;
        std::vector<char*> av = {argv[0], const_cast<char*>(queuing.c_str())};

        hpx::local::init_params init_args;
        init_args.cfg = {"hpx.os_threads=4"};

        HPX_TEST_EQ(hpx::local::init(hpx_main, static_cast<int>(av.size()),
                        av.data(), init_args),
            0);
    }

    return hpx::util::report_errors();
}

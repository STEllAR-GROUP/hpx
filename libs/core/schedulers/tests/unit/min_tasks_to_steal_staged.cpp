//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpx.thread_queue.min_tasks_to_steal_staged must only keep other workers from
// stealing staged tasks: a worker has to convert its own staged tasks even if
// there are fewer of them than that minimum (this used to hang).

#include <hpx/execution.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/latch.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

constexpr std::size_t num_tasks = 3;    // fewer than the minimum below

int hpx_main()
{
    std::size_t const num_workers = hpx::get_num_worker_threads();
    for (std::size_t worker = 0; worker != num_workers; ++worker)
    {
        std::atomic<std::size_t> count(0);
        hpx::latch done(static_cast<std::ptrdiff_t>(num_tasks + 1));

        // fire-and-forget tasks are staged on the hinted worker
        hpx::execution::parallel_executor exec(
            hpx::threads::thread_schedule_hint(
                static_cast<std::int16_t>(worker)));
        for (std::size_t i = 0; i != num_tasks; ++i)
        {
            hpx::post(exec, [&] {
                ++count;
                done.count_down(1);
            });
        }

        done.arrive_and_wait();
        HPX_TEST_EQ(count.load(), num_tasks);
    }

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    hpx::local::init_params init_args;
    init_args.cfg = {
        "hpx.os_threads=4", "hpx.thread_queue.min_tasks_to_steal_staged=10"};

    HPX_TEST_EQ(hpx::local::init(hpx_main, argc, argv, init_args), 0);
    return hpx::util::report_errors();
}

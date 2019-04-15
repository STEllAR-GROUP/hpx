//  Copyright (c) 2019 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/parallel_executor_parameters.hpp>
#include <hpx/include/parallel_for_loop.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

int hpx_main(int argc, char* argv[])
{
    using hpx::parallel::for_loop;
    using hpx::parallel::execution::chunked_placement;
    using hpx::parallel::execution::par;
    using hpx::parallel::execution::round_robin_placement;
    using hpx::parallel::execution::static_chunk_size;
    using hpx::threads::thread_schedule_hint;

    std::size_t const num_tasks = 10;
    std::size_t const num_threads = hpx::get_num_worker_threads();

    // Disable stealing to ensure that tasks stay where they are scheduled.
    hpx::threads::remove_scheduler_mode(
        hpx::threads::policies::enable_stealing);

    round_robin_placement check_round_robin_placement;
    for_loop(par.with(static_chunk_size(1), round_robin_placement()),
        std::size_t(0), num_tasks,
        [&check_round_robin_placement, num_threads](std::size_t i) {
            void* dummy_executor = nullptr;
            thread_schedule_hint const hint =
                check_round_robin_placement.get_schedule_hint(
                    dummy_executor, i, num_tasks, num_threads);
            std::size_t const expected_thread = hint.hint;
            std::size_t const actual_thread = hpx::get_worker_thread_num();

            HPX_TEST_EQ(expected_thread, actual_thread);
        });

    chunked_placement check_chunked_placement;
    for_loop(par.with(static_chunk_size(1), chunked_placement()),
        std::size_t(0), num_tasks,
        [&check_chunked_placement, num_threads](std::size_t i) {
            void* dummy_executor = nullptr;
            thread_schedule_hint const hint =
                check_chunked_placement.get_schedule_hint(
                    dummy_executor, i, num_tasks, num_threads);
            std::size_t const expected_thread = hint.hint;
            std::size_t const actual_thread = hpx::get_worker_thread_num();

            HPX_TEST_EQ(expected_thread, actual_thread);
        });

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg = {"hpx.os_threads=4"};

    HPX_TEST_EQ(hpx::init(argc, argv, cfg), 0);

    return hpx::util::report_errors();
}

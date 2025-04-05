// Copyright (c) 2024 Harith
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test verifies run_on_all functionality with different execution policies
#include <hpx/async.hpp>
#include <hpx/execution.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/parallel/algorithms/for_loop_reduction.hpp>
#include <hpx/parallel/run_on_all.hpp>
#include <hpx/testing.hpp>

#include <atomic>
#include <cstddef>
#include <vector>

int hpx_main()
{
    // We're using a single node/locality, so we need to use a number of tasks
    // equal to the number of OS threads, not the number of localities.
    const std::size_t num_tasks = hpx::get_os_thread_count();

    {
        // Test synchronous execution
        std::atomic<std::size_t> count{0};
        hpx::experimental::run_on_all(num_tasks, [&count]() { ++count; });

        HPX_TEST_EQ(count.load(), num_tasks);
    }

    {
        // Test asynchronous execution with execution policy
        std::atomic<std::size_t> count{0};
        auto futures = hpx::experimental::run_on_all(
            hpx::execution::par(hpx::execution::task), num_tasks,
            [&count]() { ++count; });

        hpx::wait_all(futures);
        HPX_TEST_EQ(count.load(), num_tasks);
    }

    {
        // Test with simple reduction approach
        std::atomic<std::size_t> sum{0};
        hpx::experimental::run_on_all(num_tasks, [&sum]() { sum += 1; });

        HPX_TEST_EQ(sum.load(), num_tasks);
    }

    {
        // Test with async reduction
        std::atomic<std::size_t> sum{0};
        auto futures = hpx::experimental::run_on_all(
            hpx::execution::par(hpx::execution::task), num_tasks,
            [&sum]() { sum += 1; });

        hpx::wait_all(futures);
        HPX_TEST_EQ(sum.load(), num_tasks);
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(
        hpx::init(argc, argv), 0, "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
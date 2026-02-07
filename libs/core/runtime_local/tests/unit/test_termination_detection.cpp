// Copyright (c) 2025 Arpit Khandelwal
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/post.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/runtime_local/termination_detection.hpp>

#include <atomic>
#include <cstddef>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Test basic local termination detection
void test_basic_termination()
{
    std::atomic<int> counter{0};

    // Launch some asynchronous work
    for (int i = 0; i < 100; ++i)
    {
        hpx::post([&counter] { ++counter; });
    }

    // Wait for all work to complete
    hpx::wait_for_local_termination();

    // Verify all work completed
    HPX_TEST_EQ(counter.load(), 100);
}

///////////////////////////////////////////////////////////////////////////////
// Test multiple sequential calls
void test_multiple_calls()
{
    std::atomic<int> counter{0};

    // First batch of work
    for (int i = 0; i < 50; ++i)
    {
        hpx::post([&counter] { ++counter; });
    }

    hpx::wait_for_local_termination();
    HPX_TEST_EQ(counter.load(), 50);

    // Second batch of work
    for (int i = 0; i < 50; ++i)
    {
        hpx::post([&counter] { ++counter; });
    }

    hpx::wait_for_local_termination();
    HPX_TEST_EQ(counter.load(), 100);
}

///////////////////////////////////////////////////////////////////////////////
// Test with nested async operations
void test_nested_async()
{
    std::atomic<int> counter{0};

    // Launch work that spawns more work
    for (int i = 0; i < 10; ++i)
    {
        hpx::post([&counter] {
            ++counter;
            // Spawn nested work
            for (int j = 0; j < 5; ++j)
            {
                hpx::post([&counter] { ++counter; });
            }
        });
    }

    // Wait for all work (including nested) to complete
    hpx::wait_for_local_termination();

    // Verify: 10 parent tasks + 50 nested tasks = 60 total
    HPX_TEST_EQ(counter.load(), 60);
}

///////////////////////////////////////////////////////////////////////////////
// Test with futures
void test_with_futures()
{
    std::vector<hpx::future<int>> futures;

    // Launch async operations that return values
    for (int i = 0; i < 20; ++i)
    {
        futures.push_back(hpx::async([] { return 42; }));
    }

    // Wait for termination
    hpx::wait_for_local_termination();

    // All futures should be ready
    for (auto& f : futures)
    {
        HPX_TEST(f.is_ready());
        HPX_TEST_EQ(f.get(), 42);
    }
}

///////////////////////////////////////////////////////////////////////////////
// Test empty workload (no pending tasks)
void test_empty_workload()
{
    // Should return immediately when no work is pending
    hpx::wait_for_local_termination();
    HPX_TEST(true);    // If we get here, the test passed
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_basic_termination();
    test_multiple_calls();
    test_nested_async();
    test_with_futures();
    test_empty_workload();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize HPX and run tests
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}

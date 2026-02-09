// Copyright (c) 2025 Arpit Khandelwal
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/post.hpp>
#include <hpx/modules/runtime_local.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <chrono>
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
    hpx::local::termination_detection();

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

    hpx::local::termination_detection();
    HPX_TEST_EQ(counter.load(), 50);

    // Second batch of work
    for (int i = 0; i < 50; ++i)
    {
        hpx::post([&counter] { ++counter; });
    }

    hpx::local::termination_detection();
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
    hpx::local::termination_detection();

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
    hpx::local::termination_detection();

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
    hpx::local::termination_detection();
    HPX_TEST(true);    // If we get here, the test passed
}

///////////////////////////////////////////////////////////////////////////////
// Test timeout with fast completion
void test_timeout_fast_completion()
{
    std::atomic<int> counter{0};

    // Launch quick work
    for (int i = 0; i < 10; ++i)
    {
        hpx::post([&counter] { ++counter; });
    }

    // Wait with generous timeout (should complete before timeout)
    bool completed = hpx::local::termination_detection(std::chrono::seconds(5));

    HPX_TEST(completed);
    HPX_TEST_EQ(counter.load(), 10);
}

///////////////////////////////////////////////////////////////////////////////
// Test timeout expiration
void test_timeout_expiration()
{
    std::atomic<bool> work_started{false};
    std::atomic<bool> work_completed{false};

    // Launch long-running work
    hpx::post([&work_started, &work_completed] {
        work_started = true;
        hpx::this_thread::sleep_for(std::chrono::seconds(2));
        work_completed = true;
    });

    // Wait for work to start
    while (!work_started.load())
    {
        hpx::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Wait with short timeout (should timeout)
    bool completed =
        hpx::local::termination_detection(std::chrono::milliseconds(100));

    HPX_TEST(!completed);                // Should timeout
    HPX_TEST(!work_completed.load());    // Work should still be running

    // Wait for actual completion
    hpx::local::termination_detection(std::chrono::seconds(5));
    HPX_TEST(work_completed.load());
}

///////////////////////////////////////////////////////////////////////////////
// Test deadline-based timeout
void test_deadline_timeout()
{
    std::atomic<int> counter{0};

    // Launch work
    for (int i = 0; i < 10; ++i)
    {
        hpx::post([&counter] { ++counter; });
    }

    // Set deadline 2 seconds from now
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);

    bool completed = hpx::local::termination_detection(deadline);

    HPX_TEST(completed);
    HPX_TEST_EQ(counter.load(), 10);
}

///////////////////////////////////////////////////////////////////////////////
// Test deadline already passed
void test_deadline_already_passed()
{
    // Set deadline in the past
    auto deadline = std::chrono::steady_clock::now() - std::chrono::seconds(1);

    bool completed = hpx::local::termination_detection(deadline);

    HPX_TEST(!completed);    // Should return false immediately
}

///////////////////////////////////////////////////////////////////////////////
// Test stop_token cancellation
void test_stop_token_cancellation()
{
    hpx::stop_source stop_src;
    std::atomic<bool> work_started{false};

    // Launch long-running work
    hpx::post([&work_started] {
        work_started = true;
        hpx::this_thread::sleep_for(std::chrono::seconds(2));
    });

    // Wait for work to start
    while (!work_started.load())
    {
        hpx::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Request stop in another thread
    hpx::thread stopper([&stop_src]() {
        hpx::this_thread::sleep_for(std::chrono::milliseconds(200));
        stop_src.request_stop();
    });

    // Wait with stop token (should be cancelled)
    bool completed = hpx::local::termination_detection(
        stop_src.get_token(), std::chrono::seconds(10));

    HPX_TEST(!completed);    // Should be cancelled

    stopper.join();

    // Clean up remaining work
    hpx::local::termination_detection(std::chrono::seconds(5));
}

///////////////////////////////////////////////////////////////////////////////
// Test stop_token with fast completion
void test_stop_token_fast_completion()
{
    hpx::stop_source stop_src;
    std::atomic<int> counter{0};

    // Launch quick work
    for (int i = 0; i < 10; ++i)
    {
        hpx::post([&counter] { ++counter; });
    }

    // Wait with stop token (should complete before any cancellation)
    bool completed = hpx::local::termination_detection(
        stop_src.get_token(), std::chrono::seconds(5));

    HPX_TEST(completed);
    HPX_TEST_EQ(counter.load(), 10);
}

///////////////////////////////////////////////////////////////////////////////
// Test timeout with zero duration
void test_zero_timeout()
{
    std::atomic<bool> work_started{false};

    // Launch work
    hpx::post([&work_started] {
        work_started = true;
        hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
    });

    // Wait for work to start
    while (!work_started.load())
    {
        hpx::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Zero timeout should return immediately
    bool completed = hpx::local::termination_detection(std::chrono::seconds(0));

    HPX_TEST(!completed);

    // Clean up
    hpx::local::termination_detection(std::chrono::seconds(5));
}

///////////////////////////////////////////////////////////////////////////////
// Test multiple timeout calls
void test_multiple_timeout_calls()
{
    std::atomic<int> counter{0};

    // First batch with timeout
    for (int i = 0; i < 25; ++i)
    {
        hpx::post([&counter] { ++counter; });
    }

    bool completed1 =
        hpx::local::termination_detection(std::chrono::seconds(2));
    HPX_TEST(completed1);
    HPX_TEST_EQ(counter.load(), 25);

    // Second batch with timeout
    for (int i = 0; i < 25; ++i)
    {
        hpx::post([&counter] { ++counter; });
    }

    bool completed2 =
        hpx::local::termination_detection(std::chrono::seconds(2));
    HPX_TEST(completed2);
    HPX_TEST_EQ(counter.load(), 50);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    // Basic tests
    test_basic_termination();
    test_multiple_calls();
    test_nested_async();
    test_with_futures();
    test_empty_workload();

    // Timeout tests
    test_timeout_fast_completion();
    test_timeout_expiration();
    test_deadline_timeout();
    test_deadline_already_passed();
    test_zero_timeout();
    test_multiple_timeout_calls();

    // Stop token tests
    test_stop_token_cancellation();
    test_stop_token_fast_completion();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize HPX and run tests
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}

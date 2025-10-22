// Copyright (c) 2025 Sai Charan Arvapally
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/execution_base/stdexec_forward.hpp>
#include <hpx/executors/parallel_scheduler.hpp>
#include <hpx/executors/thread_pool_scheduler.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>    // For thread ID
#include <chrono>
#include <exception>
#include <iostream>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

namespace ex = hpx::execution::experimental;

// Stream output operator for forward_progress_guarantee
std::ostream& operator<<(
    std::ostream& os, ex::forward_progress_guarantee guarantee)
{
    switch (guarantee)
    {
    case ex::forward_progress_guarantee::concurrent:
        return os << "concurrent";
    case ex::forward_progress_guarantee::parallel:
        return os << "parallel";
    case ex::forward_progress_guarantee::weakly_parallel:
        return os << "weakly_parallel";
    default:
        return os << "unknown";
    }
}

// Custom receiver to test completion signatures and stop token
struct test_receiver
{
#if defined(HPX_HAVE_STDEXEC)
    using receiver_concept = ex::receiver_t;
#endif

    bool* value_received;
    std::exception_ptr* error_received;
    bool* stopped_received;
    hpx::thread::id* thread_id;    // Track thread ID

    test_receiver(bool* vr, std::exception_ptr* er, bool* sr,
        hpx::thread::id* tid = nullptr)
      : value_received(vr)
      , error_received(er)
      , stopped_received(sr)
      , thread_id(tid)
    {
    }

    // Tag invoke implementations for stdexec
    friend void tag_invoke(ex::set_value_t, test_receiver&& r) noexcept
    {
        if (r.thread_id)
            *r.thread_id = hpx::this_thread::get_id();
        *r.value_received = true;
        std::cout << "set_value called on thread " << hpx::this_thread::get_id()
                  << std::endl;
    }

    friend void tag_invoke(
        ex::set_value_t, test_receiver&& r, int value) noexcept
    {
        if (r.thread_id)
            *r.thread_id = hpx::this_thread::get_id();
        *r.value_received = true;
        std::cout << "set_value(int: " << value << ") called on thread "
                  << hpx::this_thread::get_id() << std::endl;
    }

    friend void tag_invoke(
        ex::set_value_t, test_receiver&& r, std::string value) noexcept
    {
        if (r.thread_id)
            *r.thread_id = hpx::this_thread::get_id();
        *r.value_received = true;
        std::cout << "set_value(string: " << value << ") called on thread "
                  << hpx::this_thread::get_id() << std::endl;
    }

    friend void tag_invoke(
        ex::set_error_t, test_receiver&& r, std::exception_ptr ep) noexcept
    {
        *r.error_received = ep;
        std::cout << "set_error called on thread " << hpx::this_thread::get_id()
                  << std::endl;
    }

    friend void tag_invoke(ex::set_stopped_t, test_receiver&& r) noexcept
    {
        *r.stopped_received = true;
        std::cout << "set_stopped called on thread "
                  << hpx::this_thread::get_id() << std::endl;
    }

    struct env
    {
    };

    friend env tag_invoke(ex::get_env_t, const test_receiver&) noexcept
    {
        return env{};
    }
};

int hpx_main(int, char*[])
{
    std::cout << "hpx_main started" << std::endl;
    auto sched = ex::get_parallel_scheduler();
    std::cout << "Obtained parallel_scheduler" << std::endl;

    // Test parallel_scheduler construction and assignment
    {
        std::cout << "\n=== Testing Scheduler Construction ===" << std::endl;
        static_assert(!std::is_default_constructible_v<ex::parallel_scheduler>,
            "parallel_scheduler should not be default constructible");

        auto sched1 = ex::get_parallel_scheduler();
        auto sched2 = sched1;                                // Copy construct
        ex::parallel_scheduler sched3(std::move(sched2));    // Move construct
        sched2 = sched1;                                     // Copy assign
        ex::parallel_scheduler sched4 = ex::get_parallel_scheduler();
        sched4 = std::move(sched3);    // Move assign

        HPX_TEST(sched1 == sched2);
        HPX_TEST(sched2 == sched4);
        std::cout << "Scheduler equality tests passed" << std::endl;

        static_assert(
            std::is_nothrow_copy_constructible_v<ex::parallel_scheduler>,
            "copy constructor should be noexcept");
        static_assert(
            std::is_nothrow_move_constructible_v<ex::parallel_scheduler>,
            "move constructor should be noexcept");
        static_assert(std::is_nothrow_copy_assignable_v<ex::parallel_scheduler>,
            "copy assignment should be noexcept");
        static_assert(std::is_nothrow_move_assignable_v<ex::parallel_scheduler>,
            "move assignment should be noexcept");
        static_assert(noexcept(sched1 == sched2),
            "equality comparison should be noexcept");
        std::cout << "Scheduler noexcept properties verified" << std::endl;
    }

    // Test parallel_scheduler forward progress guarantee
    {
        std::cout << "\n=== Testing Forward Progress Guarantee ==="
                  << std::endl;
        auto guarantee = ex::get_forward_progress_guarantee(sched);
        std::cout << "Forward progress guarantee: " << guarantee << std::endl;
        HPX_TEST(guarantee == ex::forward_progress_guarantee::parallel);
        static_assert(noexcept(ex::get_forward_progress_guarantee(sched)),
            "get_forward_progress_guarantee should be noexcept");
        std::cout << "Forward progress guarantee test passed" << std::endl;
    }

    // Test parallel_scheduler schedule
    {
        std::cout << "\n=== Testing Schedule ===" << std::endl;
        auto sender = ex::schedule(sched);
        std::cout << "Created sender with schedule" << std::endl;
        (void) sender;    // Suppress unused variable warning

#if defined(HPX_HAVE_STDEXEC)
        static_assert(
            ex::sender<decltype(sender)>, "schedule should return a sender");

        using completion_sigs =
            ex::completion_signatures_of_t<decltype(sender), ex::env<>>;
        static_assert(
            std::is_same_v<completion_sigs,
                ex::completion_signatures<ex::set_value_t(),
                    ex::set_error_t(std::exception_ptr), ex::set_stopped_t()>>,
            "sender should have correct completion signatures");
#endif
        static_assert(
            noexcept(ex::schedule(sched)), "schedule should be noexcept");
        std::cout << "Schedule sender properties verified" << std::endl;
    }

    // Test parallel_scheduler completion scheduler query
    {
        std::cout << "\n=== Testing Completion Scheduler Query ==="
                  << std::endl;
        auto sender = ex::schedule(sched);
        auto env = ex::get_env(sender);
        auto completion_sched =
            ex::get_completion_scheduler<ex::set_value_t>(env);
        HPX_TEST(completion_sched == sched);
        static_assert(
            noexcept(ex::get_env(sender)), "get_env should be noexcept");
        static_assert(
            noexcept(ex::get_completion_scheduler<ex::set_value_t>(env)),
            "get_completion_scheduler should be noexcept");
        std::cout << "Completion scheduler query test passed" << std::endl;
    }

    // Test parallel_scheduler stop token
    {
        std::cout << "\n=== Testing Stop Token ===" << std::endl;
        bool value_received = false;
        std::exception_ptr error_received = nullptr;
        bool stopped_received = false;

        test_receiver receiver(
            &value_received, &error_received, &stopped_received, nullptr);

        auto sender = ex::schedule(sched);
        auto op_state = ex::connect(sender, receiver);
        std::cout << "Calling start for stop token test" << std::endl;
        ex::start(op_state);

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        HPX_TEST(!error_received);
        std::cout << "Stop token test: stopped_received = " << stopped_received
                  << ", error_received = " << (error_received != nullptr)
                  << std::endl;
    }

    // Test parallel_scheduler shared context
    {
        std::cout << "\n=== Testing Shared Context ===" << std::endl;
        auto sched1 = ex::get_parallel_scheduler();
        auto sched2 = ex::get_parallel_scheduler();
        HPX_TEST(sched1 == sched2);
        std::cout << "Schedulers share same context" << std::endl;
    }

#if defined(HPX_HAVE_STDEXEC)
    // Test parallel_scheduler basic execution
    {
        std::cout << "\n=== Testing Basic Execution ===" << std::endl;
        auto sender = ex::schedule(sched) | ex::then([] {
            std::cout << "Executing then functor returning 42" << std::endl;
            return 42;
        });

        std::cout << "Calling sync_wait for basic execution" << std::endl;
        auto [result] = ex::sync_wait(sender).value();
        HPX_TEST_EQ(result, 42);
        std::cout << "Basic execution result: " << result << std::endl;
    }

    // Test parallel_scheduler structured concurrency with on
    {
        std::cout << "\n=== Testing Structured Concurrency with on ==="
                  << std::endl;
        auto sender = ex::on(sched, ex::then(ex::just(), [] {
            std::cout << "Executing then functor returning 'Hello, P2079!'"
                      << std::endl;
            return std::string("Hello, P2079!");
        }));

        std::cout << "Calling sync_wait for structured concurrency"
                  << std::endl;
        auto [result] = ex::sync_wait(sender).value();
        HPX_TEST_EQ(result, std::string("Hello, P2079!"));
        std::cout << "Structured concurrency result: " << result << std::endl;
    }

    // Test parallel_scheduler error handling
    {
        std::cout << "\n=== Testing Error Handling ===" << std::endl;
        auto sender = ex::schedule(sched) | ex::then([] {
            std::cout << "Throwing runtime_error in then functor" << std::endl;
            throw std::runtime_error("Test error");
            return 0;
        });

        bool caught_error = false;
        try
        {
            std::cout << "Calling sync_wait for error handling" << std::endl;
            auto result = ex::sync_wait(sender);
            HPX_TEST(false);
        }
        catch (const std::runtime_error& e)
        {
            caught_error = true;
            HPX_TEST_EQ(std::string(e.what()), std::string("Test error"));
            std::cout << "Caught error: " << e.what() << std::endl;
        }
        HPX_TEST(caught_error);
        std::cout << "Error handling test passed" << std::endl;
    }

    // Test parallel_scheduler shared context with algorithms
    {
        std::cout << "\n=== Testing Shared Context with Algorithms ==="
                  << std::endl;
        auto sched1 = ex::get_parallel_scheduler();
        auto sched2 = ex::get_parallel_scheduler();
        HPX_TEST(sched1 == sched2);
        std::cout << "Schedulers share same context" << std::endl;

        int count1 = 0, count2 = 0;
        auto sender1 = ex::schedule(sched1) | ex::then([&count1] {
            std::cout << "Executing sender1 then functor" << std::endl;
            count1++;
        });
        auto sender2 = ex::schedule(sched2) | ex::then([&count2] {
            std::cout << "Executing sender2 then functor" << std::endl;
            count2++;
        });

        std::cout << "Calling sync_wait for shared context" << std::endl;
        ex::sync_wait(ex::when_all(sender1, sender2));
        HPX_TEST_EQ(count1, 1);
        HPX_TEST_EQ(count2, 1);
        std::cout << "Shared context test: count1 = " << count1
                  << ", count2 = " << count2 << std::endl;
    }

    // Test parallel_scheduler example from P2079R10
    {
        std::cout << "\n=== Testing P2079R10 Examples ===" << std::endl;
        auto begin = ex::schedule(sched);
        auto hi = ex::then(begin, [] {
            std::cout << "Executing P2079R10 Example 1 then functor"
                      << std::endl;
            return 13;
        });
        auto add_42 = ex::then(hi, [](int arg) {
            std::cout << "Adding 42 to " << arg << std::endl;
            return arg + 42;
        });
        std::cout << "Calling sync_wait for P2079R10 Example 1" << std::endl;
        auto [i] = ex::sync_wait(add_42).value();
        HPX_TEST_EQ(i, 55);
        std::cout << "P2079R10 Example 1 result: " << i << std::endl;

        auto hi2 = ex::then(ex::just(), [] {
            std::cout << "Executing P2079R10 Example 2 then functor"
                      << std::endl;
            return 13;
        });
        auto add_42_2 = ex::then(hi2, [](int arg) {
            std::cout << "Adding 42 to " << arg << std::endl;
            return arg + 42;
        });
        std::cout << "Calling sync_wait for P2079R10 Example 2" << std::endl;
        auto [i2] = ex::sync_wait(ex::on(sched, add_42_2)).value();
        HPX_TEST_EQ(i2, 55);
        std::cout << "P2079R10 Example 2 result: " << i2 << std::endl;
    }

    // Test case 1: Verify HPX thread ID for async task
    {
        std::cout << "\n=== Test Case 1: Verify HPX Thread ID ===" << std::endl;
        bool value_received = false;
        std::exception_ptr error_received = nullptr;
        bool stopped_received = false;
        hpx::thread::id thread_id;

        test_receiver receiver(
            &value_received, &error_received, &stopped_received, &thread_id);
        auto sender = ex::schedule(sched) | ex::then([] {
            std::cout << "Executing Test Case 1 then functor" << std::endl;
            return 1;
        });

        std::cout << "Calling connect and start for Test Case 1" << std::endl;
        auto op_state = ex::connect(sender, receiver);
        ex::start(op_state);

        std::cout << "Waiting for Test Case 1 to complete" << std::endl;
        while (!value_received && !error_received && !stopped_received)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        HPX_TEST(value_received);
        HPX_TEST(thread_id != hpx::thread::id{});
        std::cout << "Test Case 1 thread ID: " << thread_id << std::endl;
    }

    // Test case 2: Verify different HPX thread ID for another async task
    {
        std::cout << "\n=== Test Case 2: Verify Different HPX Thread ID ==="
                  << std::endl;
        bool value_received = false;
        std::exception_ptr error_received = nullptr;
        bool stopped_received = false;
        hpx::thread::id thread_id;

        test_receiver receiver(
            &value_received, &error_received, &stopped_received, &thread_id);
        auto sender = ex::schedule(sched) | ex::then([] {
            std::cout << "Executing Test Case 2 then functor" << std::endl;
            return 2;
        });

        std::cout << "Calling connect and start for Test Case 2" << std::endl;
        auto op_state = ex::connect(sender, receiver);
        ex::start(op_state);

        std::cout << "Waiting for Test Case 2 to complete" << std::endl;
        while (!value_received && !error_received && !stopped_received)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        HPX_TEST(value_received);
        HPX_TEST(thread_id != hpx::thread::id{});
        std::cout << "Test Case 2 thread ID: " << thread_id << std::endl;
    }
#endif    // HPX_HAVE_STDEXEC

    // Test bulk execution with parallel scheduler
    {
        std::cout << "\n=== Testing Bulk Execution ===" << std::endl;
        const std::size_t n = 100;
        std::atomic<std::size_t> counter{0};
        std::mutex info_mutex;
        std::map<hpx::thread::id, std::vector<std::size_t>> thread_work_map;
        std::vector<std::pair<std::size_t, hpx::thread::id>> execution_order;

        auto bulk_sender =
            ex::schedule(sched) | ex::bulk(n, [&](std::size_t i) {
                auto tid = hpx::this_thread::get_id();
                counter.fetch_add(1, std::memory_order_relaxed);

                {
                    std::lock_guard<std::mutex> lock(info_mutex);
                    thread_work_map[tid].push_back(i);
                    execution_order.emplace_back(i, tid);
                }

                // Add some work to potentially trigger multi-threading
                volatile double dummy = 0.0;
                for (int j = 0; j < 1000; ++j)
                {
                    dummy += std::sin(i * j * 0.001);
                }
            });

        std::cout << "Calling sync_wait for bulk execution" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        ex::sync_wait(bulk_sender);
        auto end_time = std::chrono::high_resolution_clock::now();

        HPX_TEST_EQ(counter.load(), n);
        std::cout << "Bulk execution completed: counter = " << counter.load()
                  << ", expected = " << n << std::endl;

        // Display detailed execution information
        std::cout << "\nExecution Details:" << std::endl;
        std::cout << "Total execution time: "
                  << std::chrono::duration<double, std::milli>(
                         end_time - start_time)
                         .count()
                  << " ms" << std::endl;

        std::cout << "\nThread-wise work distribution:" << std::endl;
        for (const auto& [tid, indices] : thread_work_map)
        {
            std::cout << "Thread " << tid << " executed " << indices.size()
                      << " items: ";

            // Find ranges of consecutive indices
            if (!indices.empty())
            {
                std::vector<std::pair<std::size_t, std::size_t>> ranges;
                std::size_t start = indices[0];
                std::size_t end = indices[0];

                for (std::size_t i = 1; i < indices.size(); ++i)
                {
                    if (indices[i] == end + 1)
                    {
                        end = indices[i];
                    }
                    else
                    {
                        ranges.emplace_back(start, end);
                        start = end = indices[i];
                    }
                }
                ranges.emplace_back(start, end);

                // Print ranges
                for (std::size_t i = 0; i < ranges.size(); ++i)
                {
                    if (i > 0)
                        std::cout << ", ";
                    if (ranges[i].first == ranges[i].second)
                    {
                        std::cout << ranges[i].first;
                    }
                    else
                    {
                        std::cout << "[" << ranges[i].first << "-"
                                  << ranges[i].second << "]";
                    }
                }
            }
            std::cout << std::endl;
        }

        std::cout << "Number of unique threads used: " << thread_work_map.size()
                  << std::endl;

        // Show first few execution order items
        std::cout << "\nFirst 10 executions:" << std::endl;
        for (std::size_t i = 0;
            i < std::min<std::size_t>(10, execution_order.size()); ++i)
        {
            std::cout << "  Index " << execution_order[i].first << " on thread "
                      << execution_order[i].second << std::endl;
        }
    }

    // Test bulk with value propagation
    {
        std::cout << "\n=== Testing Bulk with Value Propagation ==="
                  << std::endl;
        const std::size_t n = 50;
        std::atomic<int> sum{0};
        std::mutex info_mutex;
        std::map<hpx::thread::id, int> thread_sum_map;

        auto bulk_sender = ex::just(10) |    // Start with value 10
            ex::bulk(n, [&](std::size_t i, int base_value) {
                auto tid = hpx::this_thread::get_id();
                sum.fetch_add(base_value, std::memory_order_relaxed);

                {
                    std::lock_guard<std::mutex> lock(info_mutex);
                    thread_sum_map[tid] += base_value;
                }
            });

        std::cout << "Calling sync_wait for bulk with value" << std::endl;
        ex::sync_wait(bulk_sender);

        HPX_TEST_EQ(sum.load(), static_cast<int>(n * 10));
        std::cout << "Bulk with value completed: sum = " << sum.load()
                  << ", expected = " << (n * 10) << std::endl;

        std::cout << "Per-thread contributions:" << std::endl;
        for (const auto& [tid, thread_sum] : thread_sum_map)
        {
            std::cout << "  Thread " << tid << " contributed: " << thread_sum
                      << std::endl;
        }
    }

    // Test small bulk operation (edge case)
    {
        std::cout << "\n=== Testing Small Bulk Operation ===" << std::endl;
        const std::size_t n = 4;    // Very small to test edge cases
        std::atomic<std::size_t> executions{0};
        std::mutex info_mutex;

        auto bulk_sender =
            ex::schedule(sched) | ex::bulk(n, [&](std::size_t i) {
                auto tid = hpx::this_thread::get_id();
                {
                    std::lock_guard<std::mutex> lock(info_mutex);
                    std::cout << "Small bulk executing index " << i
                              << " on thread " << tid << std::endl;
                }
                executions.fetch_add(1, std::memory_order_relaxed);
            });

        std::cout << "Calling sync_wait for small bulk operation" << std::endl;
        ex::sync_wait(bulk_sender);

        HPX_TEST_EQ(executions.load(), n);
        std::cout << "Small bulk completed with " << executions.load()
                  << " executions" << std::endl;
    }

    // Test bulk error handling
    {
        std::cout << "\n=== Testing Bulk Error Handling ===" << std::endl;
        const std::size_t n = 20;
        bool caught_error = false;
        std::atomic<std::size_t> executed_before_error{0};

        auto bulk_sender =
            ex::schedule(sched) | ex::bulk(n, [&](std::size_t i) {
                if (i < 10)
                {
                    executed_before_error.fetch_add(
                        1, std::memory_order_relaxed);
                }
                if (i == 10)
                {    // Throw on a specific index
                    std::cout << "Throwing error from index " << i
                              << " on thread " << hpx::this_thread::get_id()
                              << std::endl;
                    throw std::runtime_error("Bulk error");
                }
            });

        try
        {
            std::cout << "Calling sync_wait for bulk error test" << std::endl;
            ex::sync_wait(bulk_sender);
            HPX_TEST(false);    // Should not reach here
        }
        catch (const std::runtime_error& e)
        {
            caught_error = true;
            std::cout << "Caught expected error: " << e.what() << std::endl;
            std::cout << "Executed " << executed_before_error.load()
                      << " items before error" << std::endl;
        }

        HPX_TEST(caught_error);
    }

    std::cout << "Calling hpx::local::finalize()" << std::endl;
    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    std::cout << "main() started" << std::endl;
    std::cout << "Calling hpx::local::init" << std::endl;
    int result = hpx::local::init(hpx_main, argc, argv);
    std::cout << "hpx::local::init returned: " << result << std::endl;
    return result;
}

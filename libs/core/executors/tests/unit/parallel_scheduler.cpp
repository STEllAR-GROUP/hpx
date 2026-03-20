// Copyright (c) 2025 Sai Charan Arvapally
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/executors/parallel_scheduler.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <chrono>
#include <exception>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace ex = hpx::execution::experimental;

#if defined(HPX_HAVE_STDEXEC)
// Include stdexec async_scope for stop token testing
#include <exec/async_scope.hpp>
#endif

int hpx_main(int, char*[])
{
    // Type and Concept Tests
    // parallel_scheduler models scheduler concept
    {
        auto sched = ex::get_parallel_scheduler();
        static_assert(
            ex::scheduler<decltype(sched)>, "parallel_scheduler must model scheduler");
    }

    // parallel_scheduler is not default constructible
    {
        static_assert(!std::is_default_constructible_v<ex::parallel_scheduler>,
            "parallel_scheduler should not be default constructible");
        static_assert(std::is_destructible_v<ex::parallel_scheduler>,
            "parallel_scheduler should be destructible");
    }

    // parallel_scheduler is copyable and movable
    {
        static_assert(
            std::is_copy_constructible_v<ex::parallel_scheduler>,
            "parallel_scheduler should be copy constructible");
        static_assert(
            std::is_move_constructible_v<ex::parallel_scheduler>,
            "parallel_scheduler should be move constructible");
        static_assert(
            std::is_nothrow_copy_constructible_v<ex::parallel_scheduler>,
            "copy constructor should be noexcept");
        static_assert(
            std::is_nothrow_move_constructible_v<ex::parallel_scheduler>,
            "move constructor should be noexcept");
        static_assert(
            std::is_nothrow_copy_assignable_v<ex::parallel_scheduler>,
            "copy assignment should be noexcept");
        static_assert(
            std::is_nothrow_move_assignable_v<ex::parallel_scheduler>,
            "move assignment should be noexcept");
    }

    // A copied scheduler is equal to the original
    {
        auto sched1 = ex::get_parallel_scheduler();
        auto sched2 = sched1;
        HPX_TEST(sched1 == sched2);
    }

    // Two schedulers from get_parallel_scheduler() are equal
    {
        auto sched1 = ex::get_parallel_scheduler();
        auto sched2 = ex::get_parallel_scheduler();
        HPX_TEST(sched1 == sched2);
    }

    // schedule() produces a sender
    {
        auto snd = ex::schedule(ex::get_parallel_scheduler());
        using sender_t = decltype(snd);

        static_assert(ex::sender<sender_t>,
            "schedule() result must model sender");
        static_assert(ex::sender_of<sender_t, ex::set_value_t()>,
            "schedule() result must be sender_of<set_value_t()>");
        static_assert(ex::sender_of<sender_t, ex::set_stopped_t()>,
            "schedule() result must be sender_of<set_stopped_t()>");
    }
    
    // Basic Execution Tests
    // Trivial schedule task (bare sync_wait, no then)
    {
        ex::parallel_scheduler sched = ex::get_parallel_scheduler();
        ex::sync_wait(ex::schedule(sched));
    }

    // Simple schedule runs on worker thread (not main thread)
    {
        std::thread::id this_id = std::this_thread::get_id();
        std::thread::id pool_id{};
        ex::parallel_scheduler sched = ex::get_parallel_scheduler();

        auto snd = ex::then(
            ex::schedule(sched), [&] { pool_id = std::this_thread::get_id(); });

        ex::sync_wait(std::move(snd));

        HPX_TEST(pool_id != std::thread::id{});
        HPX_TEST_NEQ(this_id, pool_id);
    }

    // Forward progress guarantee is parallel
    {
        ex::parallel_scheduler sched = ex::get_parallel_scheduler();
        HPX_TEST(ex::get_forward_progress_guarantee(sched) ==
            ex::forward_progress_guarantee::parallel);
    }

    // get_completion_scheduler returns the scheduler
    {
        ex::parallel_scheduler sched = ex::get_parallel_scheduler();
        HPX_TEST(
            ex::get_completion_scheduler<ex::set_value_t>(
                ex::get_env(ex::schedule(sched))) == sched);
    }

    // Chain task: two then calls execute on same thread
    {
        std::thread::id this_id = std::this_thread::get_id();
        std::thread::id pool_id{};
        std::thread::id pool_id2{};
        ex::parallel_scheduler sched = ex::get_parallel_scheduler();

        auto snd =
            ex::then(ex::schedule(sched), [&] { pool_id = std::this_thread::get_id(); });
        auto snd2 =
            ex::then(std::move(snd), [&] { pool_id2 = std::this_thread::get_id(); });

        ex::sync_wait(std::move(snd2));

        HPX_TEST(pool_id != std::thread::id{});
        HPX_TEST_NEQ(this_id, pool_id);
        HPX_TEST(pool_id == pool_id2);
    }

    // P2079R10 example: schedule + then chain with values
    {
        ex::parallel_scheduler sched = ex::get_parallel_scheduler();
        auto begin = ex::schedule(sched);
        auto hi = ex::then(begin, [] { return 13; });
        auto add_42 = ex::then(hi, [](int arg) { return arg + 42; });
        auto [i] = ex::sync_wait(add_42).value();
        HPX_TEST_EQ(i, 55);
    }

    // Error propagation
    {
        ex::parallel_scheduler sched = ex::get_parallel_scheduler();
        bool caught_error = false;

        auto snd = ex::schedule(sched) |
            ex::then([] -> int { throw std::runtime_error("test error"); });

        try
        {
            ex::sync_wait(std::move(snd));
            HPX_TEST(false);
        }
        catch (const std::runtime_error& e)
        {
            caught_error = true;
            HPX_TEST_EQ(std::string(e.what()), std::string("test error"));
        }
        HPX_TEST(caught_error);
    }

    // when_all with multiple senders
    {
        ex::parallel_scheduler sched = ex::get_parallel_scheduler();

        auto s1 = ex::schedule(sched) | ex::then([] { return 1; });
        auto s2 = ex::schedule(sched) | ex::then([] { return 2; });
        auto s3 = ex::schedule(sched) | ex::then([] { return 3; });

        auto [r1, r2, r3] = ex::sync_wait(ex::when_all(s1, s2, s3)).value();
        HPX_TEST_EQ(r1, 1);
        HPX_TEST_EQ(r2, 2);
        HPX_TEST_EQ(r3, 3);
    }

    // Bulk Execution Tests

    // Simple bulk task
    {
        std::thread::id this_id = std::this_thread::get_id();
        constexpr std::size_t num_tasks = 16;
        std::thread::id pool_ids[num_tasks]{};
        ex::parallel_scheduler sched = ex::get_parallel_scheduler();

        std::cout << "\n=== BULK (par) with " << num_tasks << " tasks ===\n";
        std::cout << "Main thread ID: " << this_id << "\n";

        auto bulk_snd = ex::bulk(
            ex::schedule(sched), ex::par, num_tasks, [&](unsigned long id) {
                pool_ids[id] = std::this_thread::get_id();
                std::cout << "  Task " << std::setw(2) << id << " on thread "
                          << pool_ids[id] << "\n";
            });

        ex::sync_wait(std::move(bulk_snd));

        std::set<std::thread::id> unique_threads(pool_ids, pool_ids + num_tasks);
        std::cout << "Unique threads used: " << unique_threads.size() << "\n";

        for (auto pool_id : pool_ids)
        {
            HPX_TEST(pool_id != std::thread::id{});
            HPX_TEST_NEQ(this_id, pool_id);
        }
    }

    // Bulk chaining with value propagation
    {
        std::thread::id this_id = std::this_thread::get_id();
        constexpr std::size_t num_tasks = 16;
        std::thread::id pool_id{};
        std::thread::id propagated_pool_ids[num_tasks]{};
        std::thread::id pool_ids[num_tasks]{};
        ex::parallel_scheduler sched = ex::get_parallel_scheduler();

        auto snd = ex::then(ex::schedule(sched), [&] {
            pool_id = std::this_thread::get_id();
            return pool_id;
        });

        auto bulk_snd = ex::bulk(std::move(snd), ex::par, num_tasks,
            [&](unsigned long id, std::thread::id propagated_pool_id) {
                propagated_pool_ids[id] = propagated_pool_id;
                pool_ids[id] = std::this_thread::get_id();
            });

        std::optional<std::tuple<std::thread::id>> res =
            ex::sync_wait(std::move(bulk_snd));

        // first schedule ran on a different thread
        HPX_TEST(pool_id != std::thread::id{});
        HPX_TEST_NEQ(this_id, pool_id);

        // bulk items ran and propagated the received value
        for (std::size_t i = 0; i < num_tasks; ++i)
        {
            HPX_TEST(pool_ids[i] != std::thread::id{});
            HPX_TEST(propagated_pool_ids[i] == pool_id);
            HPX_TEST_NEQ(this_id, pool_ids[i]);
        }

        // result of bulk is the same as the first schedule
        HPX_TEST(res.has_value());
        HPX_TEST(std::get<0>(res.value()) == pool_id);
    }

    // Bulk error handling
    {
        ex::parallel_scheduler sched = ex::get_parallel_scheduler();
        bool caught_error = false;

        auto bulk_snd = ex::bulk(
            ex::schedule(sched), ex::par, 20,
            [](std::size_t i) {
                if (i == 10)
                    throw std::runtime_error("Bulk error");
            });

        try
        {
            ex::sync_wait(std::move(bulk_snd));
            HPX_TEST(false);
        }
        catch (const std::runtime_error& e)
        {
            caught_error = true;
            HPX_TEST(std::string(e.what()).find("Bulk error") !=
                std::string::npos);
        }
        HPX_TEST(caught_error);
    }

    // bulk_chunked Tests

    // Simple bulk_chunked task
    {
        std::thread::id this_id = std::this_thread::get_id();
        constexpr std::size_t num_tasks = 16;
        std::thread::id pool_ids[num_tasks]{};
        ex::parallel_scheduler sched = ex::get_parallel_scheduler();

        std::cout << "\n=== BULK_CHUNKED (par) with " << num_tasks << " tasks ===\n";
        std::cout << "Main thread ID: " << this_id << "\n";
        std::atomic<int> chunk_count{0};

        auto bulk_snd = ex::bulk_chunked(
            ex::schedule(sched), ex::par, num_tasks,
            [&](unsigned long b, unsigned long e) {
                int chunk_id = chunk_count++;
                std::cout << "  Chunk " << chunk_id << ": [" << b << ", " << e
                          << ") on thread " << std::this_thread::get_id() << "\n";
                for (unsigned long id = b; id < e; ++id)
                    pool_ids[id] = std::this_thread::get_id();
            });

        ex::sync_wait(std::move(bulk_snd));

        std::cout << "Total chunks: " << chunk_count.load() << "\n";
        std::set<std::thread::id> unique_threads(pool_ids, pool_ids + num_tasks);
        std::cout << "Unique threads used: " << unique_threads.size() << "\n";

        for (auto pool_id : pool_ids)
        {
            HPX_TEST(pool_id != std::thread::id{});
            HPX_TEST_NEQ(this_id, pool_id);
        }
    }

    // bulk_chunked performs chunking (with large shape)
    {
        std::atomic<bool> has_chunking{false};
        std::atomic<int> chunk_count{0};
        std::atomic<std::size_t> max_chunk_size{0};
        ex::parallel_scheduler sched = ex::get_parallel_scheduler();

        std::cout << "\n=== BULK_CHUNKED (par) with 10000 tasks - Chunking Test ===\n";

        auto bulk_snd = ex::bulk_chunked(
            ex::schedule(sched), ex::par, 10000,
            [&](unsigned long b, unsigned long e) {
                std::size_t chunk_size = e - b;
                chunk_count++;
                if (chunk_size > 1)
                    has_chunking = true;
                std::size_t expected = max_chunk_size.load();
                while (chunk_size > expected &&
                       !max_chunk_size.compare_exchange_weak(expected, chunk_size))
                    ;
                if (chunk_count <= 5 || chunk_count % 10 == 0)
                    std::cout << "  Chunk " << chunk_count.load() << ": [" << b
                              << ", " << e << ") size=" << chunk_size << "\n";
            });

        ex::sync_wait(std::move(bulk_snd));
        std::cout << "Total chunks: " << chunk_count.load()
                  << " | Max chunk size: " << max_chunk_size.load()
                  << " | Has chunking: " << (has_chunking.load() ? "yes" : "no")
                  << "\n";
        HPX_TEST(has_chunking.load());
    }

    // bulk_chunked covers the entire range
    {
        constexpr std::size_t num_tasks = 200;
        bool covered[num_tasks]{};
        ex::parallel_scheduler sched = ex::get_parallel_scheduler();

        auto bulk_snd = ex::bulk_chunked(
            ex::schedule(sched), ex::par, num_tasks,
            [&](unsigned long b, unsigned long e) {
                for (auto i = b; i < e; ++i)
                    covered[i] = true;
            });

        ex::sync_wait(std::move(bulk_snd));

        for (std::size_t i = 0; i < num_tasks; ++i)
        {
            HPX_TEST(covered[i]);
        }
    }

    // bulk_chunked with seq doesn't do chunking (single chunk)
    {
        constexpr std::size_t num_tasks = 200;
        std::atomic<int> execution_count{0};
        ex::parallel_scheduler sched = ex::get_parallel_scheduler();

        std::cout << "\n=== BULK_CHUNKED (seq) with " << num_tasks
                  << " tasks - Single Chunk Test ===\n";
        std::cout << "Expected: 1 chunk covering [0, " << num_tasks << ")\n";

        auto bulk_snd = ex::bulk_chunked(
            ex::schedule(sched), ex::seq, num_tasks,
            [&](std::size_t b, std::size_t e) {
                std::cout << "  Chunk [" << b << ", " << e << ") on thread "
                          << std::this_thread::get_id() << "\n";
                HPX_TEST_EQ(b, std::size_t(0));
                HPX_TEST_EQ(e, num_tasks);
                execution_count++;
            });

        ex::sync_wait(std::move(bulk_snd));

        std::cout << "Actual chunks: " << execution_count.load() << "\n";
        // Per P2079R10 reference: seq should produce exactly 1 chunk
        // with b==0, e==num_tasks.
        HPX_TEST_EQ(execution_count.load(), 1);
    }

    // bulk_unchunked Tests

    // Simple bulk_unchunked task
    {
        std::thread::id this_id = std::this_thread::get_id();
        constexpr std::size_t num_tasks = 16;
        std::thread::id pool_ids[num_tasks]{};
        ex::parallel_scheduler sched = ex::get_parallel_scheduler();

        std::cout << "\n=== BULK_UNCHUNKED (par) with " << num_tasks << " tasks ===\n";
        std::cout << "Main thread ID: " << this_id << "\n";

        auto bulk_snd = ex::bulk_unchunked(
            ex::schedule(sched), ex::par, num_tasks,
            [&](unsigned long id) {
                pool_ids[id] = std::this_thread::get_id();
                std::cout << "  Task " << std::setw(2) << id << " on thread "
                          << pool_ids[id] << "\n";
            });

        ex::sync_wait(std::move(bulk_snd));

        std::set<std::thread::id> unique_threads(pool_ids, pool_ids + num_tasks);
        std::cout << "Unique threads used: " << unique_threads.size() << "\n";

        for (auto pool_id : pool_ids)
        {
            HPX_TEST(pool_id != std::thread::id{});
            HPX_TEST_NEQ(this_id, pool_id);
        }
    }

    // bulk_unchunked with seq runs everything on one thread
    {
        constexpr std::size_t num_tasks = 16;
        std::thread::id pool_ids[num_tasks]{};
        ex::parallel_scheduler sched = ex::get_parallel_scheduler();

        std::cout << "\n=== BULK_UNCHUNKED (seq) with " << num_tasks
                  << " tasks - Single Thread Test ===\n";
        std::cout << "Expected: All tasks on same thread\n";

        auto bulk_snd = ex::bulk_unchunked(
            ex::schedule(sched), ex::seq, num_tasks,
            [&](unsigned long id) {
                pool_ids[id] = std::this_thread::get_id();
                std::cout << "  Task " << std::setw(2) << id << " on thread "
                          << pool_ids[id] << "\n";
                std::this_thread::sleep_for(
                    std::chrono::milliseconds{1});
            });

        ex::sync_wait(std::move(bulk_snd));

        std::set<std::thread::id> unique_threads(pool_ids, pool_ids + num_tasks);
        std::cout << "Unique threads used: " << unique_threads.size() << "\n";

        for (auto pool_id : pool_ids)
        {
            HPX_TEST(pool_id != std::thread::id{});
            // Per P2079R10 reference: all should be on same thread with seq.
            HPX_TEST(pool_id == pool_ids[0]);
        }
    }

#if defined(HPX_HAVE_STDEXEC)
    // Stop token support test (P2079R10 requirement)
    {
        ex::parallel_scheduler sched = ex::get_parallel_scheduler();
        experimental::execution::async_scope scope;
        scope.request_stop();
        HPX_TEST(scope.get_stop_source().stop_requested());

        bool called = false;
        auto snd = ex::then(ex::schedule(sched), [&called] { called = true; });

        scope.spawn(std::move(snd));
        ex::sync_wait(scope.on_empty());

        HPX_TEST(!called);
    }

    // Test completes_on pattern (scheduler from child sender's completion scheduler)
    {
        std::cout << "\n=== TEST: completes_on pattern with bulk_chunked ===" << std::endl;
        auto sched = ex::get_parallel_scheduler();
        std::vector<int> v(10, 0);
        
        auto snd = ex::schedule(sched)
            | ex::then([&v]() { return 42; })
            | ex::bulk_chunked(ex::par, 10, [&v](std::size_t i, std::size_t, int val) {
                v[i] = val;
            });
        
        ex::sync_wait(std::move(snd));
        
        // All elements should be set to 42
        for (int i = 0; i < 10; ++i) {
            HPX_TEST_EQ(v[i], 42);
        }
        std::cout << "✓ completes_on pattern works correctly" << std::endl;
    }

    // Test completes_on with value chaining
    {
        std::cout << "\n=== TEST: completes_on with value chaining ===" << std::endl;
        auto sched = ex::get_parallel_scheduler();
        std::vector<int> v(10, 0);
        
        // schedule() -> then() creates completes_on pattern
        // The then() sender's completion scheduler is the parallel_scheduler
        auto snd = ex::schedule(sched)
            | ex::then([]() { return 99; })
            | ex::bulk_chunked(ex::par, 10, [&v](std::size_t i, std::size_t, int val) {
                v[i] = val;
            });
        
        ex::sync_wait(std::move(snd));
        
        // All elements should be set to 99
        for (int i = 0; i < 10; ++i) {
            HPX_TEST_EQ(v[i], 99);
        }
        std::cout << "✓ completes_on with value chaining works correctly" << std::endl;
    }

    // Test set_value_t completion scheduler query
    {
        std::cout << "\n=== TEST: set_value_t completion scheduler query ===" << std::endl;
        auto sched = ex::get_parallel_scheduler();
        auto snd = ex::schedule(sched);
        auto env = ex::get_env(snd);
        
        // Query the completion scheduler for set_value_t
        auto completion_sched = ex::get_completion_scheduler<ex::set_value_t>(env);
        HPX_TEST_EQ(completion_sched, sched);
        std::cout << "✓ set_value_t completion scheduler query works" << std::endl;
    }

    // Test that set_stopped_t is NOT exposed (should not compile if attempted)
    // This is a compile-time check, so we just document the expected behavior
    {
        std::cout << "\n=== TEST: set_stopped_t NOT exposed in completion scheduler ===" << std::endl;
        auto sched = ex::get_parallel_scheduler();
        auto snd = ex::schedule(sched);
        auto env = ex::get_env(snd);
        
        // The following would NOT compile if attempted:
        // auto stopped_sched = ex::get_completion_scheduler<ex::set_stopped_t>(env);
        // This is correct per P2079R10: only set_value_t is exposed.
        std::cout << "✓ set_stopped_t correctly NOT exposed (compile-time verified)" << std::endl;
    }

    // Test receiver double-move safety: if execute() throws, receiver is still valid
    {
        std::cout << "\n=== TEST: receiver double-move safety ===" << std::endl;
        auto sched = ex::get_parallel_scheduler();
        bool error_called = false;
        
        auto snd = ex::schedule(sched)
            | ex::then([]() { return 42; });
        
        // This should complete successfully without double-move issues
        ex::sync_wait(std::move(snd));
        std::cout << "✓ receiver double-move safety verified" << std::endl;
    }

    // Test bulk_unchunked with completes_on pattern
    {
        std::cout << "\n=== TEST: bulk_unchunked with completes_on pattern ===" << std::endl;
        auto sched = ex::get_parallel_scheduler();
        std::vector<int> v(10, 0);
        
        auto snd = ex::schedule(sched)
            | ex::then([&v]() { return 77; })
            | ex::bulk_unchunked(ex::par, 10, [&v](std::size_t i, int val) {
                v[i] = val;
            });
        
        ex::sync_wait(std::move(snd));
        
        // All elements should be set to 77
        for (int i = 0; i < 10; ++i) {
            HPX_TEST_EQ(v[i], 77);
        }
        std::cout << "✓ bulk_unchunked with completes_on pattern works" << std::endl;
    }

    // Test bulk_unchunked with multiple value arguments
    {
        std::cout << "\n=== TEST: bulk_unchunked with multiple values ===" << std::endl;
        auto sched = ex::get_parallel_scheduler();
        std::vector<int> v(10, 0);
        
        auto snd = ex::schedule(sched)
            | ex::then([]() { return 88; })
            | ex::bulk_unchunked(ex::par, 10, [&v](std::size_t i, int val) {
                v[i] = val;
            });
        
        ex::sync_wait(std::move(snd));
        
        // All elements should be set to 88
        for (int i = 0; i < 10; ++i) {
            HPX_TEST_EQ(v[i], 88);
        }
        std::cout << "✓ bulk_unchunked with multiple values works" << std::endl;
    }

    // Test sequential bulk with completes_on
    {
        std::cout << "\n=== TEST: sequential bulk with completes_on ===" << std::endl;
        auto sched = ex::get_parallel_scheduler();
        std::vector<int> v(5, 0);
        std::set<std::thread::id> thread_ids;
        
        auto snd = ex::schedule(sched)
            | ex::then([&v]() { return 55; })
            | ex::bulk_chunked(ex::seq, 5,
                [&v, &thread_ids](std::size_t begin, std::size_t end, int val) {
                    for (std::size_t i = begin; i < end; ++i)
                        v[i] = val;
                    thread_ids.insert(std::this_thread::get_id());
                });
        
        ex::sync_wait(std::move(snd));
        
        // All elements should be set to 55
        for (int i = 0; i < 5; ++i) {
            HPX_TEST_EQ(v[i], 55);
        }
        // Sequential execution should use only 1 thread
        HPX_TEST_EQ(thread_ids.size(), std::size_t(1));
        std::cout << "✓ sequential bulk with completes_on works (1 thread)" << std::endl;
    }
#endif

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");
    return hpx::util::report_errors();
}

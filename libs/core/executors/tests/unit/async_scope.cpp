//  Copyright (c) 2026 Sai Charan Arvapally
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Tests for P3149 async_scope facilities (spawn, spawn_future, associate)

#include <hpx/config.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/executors.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <chrono>
#include <exception>
#include <semaphore>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <hpx/modules/execution_base.hpp>

namespace ex = hpx::execution::experimental;

// spawn_future with thread_pool_scheduler: value propagation
void test_spawn_future_value()
{
    ex::simple_counting_scope scope;

    auto fut = ex::spawn_future(ex::schedule(ex::thread_pool_scheduler{}) |
            ex::then([]() { return 42; }),
        scope.get_token());

    scope.close();

    auto result = ex::sync_wait(std::move(fut));
    HPX_TEST(result.has_value());
    auto [val] = std::move(*result);
    HPX_TEST_EQ(val, 42);

    ex::sync_wait(scope.join());
}

// spawn_future with thread_pool_scheduler: error propagation
void test_spawn_future_error()
{
    ex::simple_counting_scope scope;

    auto fut = ex::spawn_future(ex::schedule(ex::thread_pool_scheduler{}) |
            ex::then([]() -> int { throw std::runtime_error("test error"); }),
        scope.get_token());

    scope.close();

    bool caught = false;
    auto handled = std::move(fut) | ex::let_error([&](auto eptr) {
        try
        {
            std::rethrow_exception(eptr);
        }
        catch (std::runtime_error const& e)
        {
            caught = true;
            HPX_TEST_EQ(std::string(e.what()), std::string("test error"));
        }
        return ex::just(-1);
    });

    auto result = ex::sync_wait(std::move(handled));
    HPX_TEST(caught);
    HPX_TEST(result.has_value());
    auto [val] = std::move(*result);
    HPX_TEST_EQ(val, -1);

    ex::sync_wait(scope.join());
}

// spawn with thread_pool_scheduler: observable side effect.
// thread_pool_scheduler's sender includes set_error_t in its completions,
// so we pipe through upon_error to satisfy spawn's no-error requirement.
void test_spawn_with_scheduler()
{
    ex::simple_counting_scope scope;
    std::atomic<int> completed{0};
    constexpr int n = 10;

    for (int i = 0; i < n; ++i)
    {
        ex::spawn(ex::schedule(ex::thread_pool_scheduler{}) |
                ex::then([&]() noexcept {
                    completed.fetch_add(1, std::memory_order_relaxed);
                }) |
                ex::upon_error([](auto) noexcept {}),
            scope.get_token());
    }

    scope.close();
    ex::sync_wait(scope.join());

    HPX_TEST_EQ(completed.load(), n);
}

// join() blocks until in-flight work on the HPX thread pool completes.
// A semaphore holds one operation; join() must not return until released.
void test_join_blocks_for_async_work()
{
    ex::simple_counting_scope scope;
    std::atomic<int> completed{0};
    constexpr int n = 8;

    std::binary_semaphore arrived{0};
    std::binary_semaphore release{0};

    for (int i = 0; i < n; ++i)
    {
        ex::spawn(ex::schedule(ex::thread_pool_scheduler{}) |
                ex::then([&, i]() noexcept {
                    if (i == 0)
                    {
                        arrived.release();
                        release.acquire();
                    }
                    completed.fetch_add(1, std::memory_order_release);
                }) |
                ex::upon_error([](auto) noexcept {}),
            scope.get_token());
    }

    // Wait until the held operation is running
    arrived.acquire();

    scope.close();

    // Start join() on a separate thread
    std::atomic<bool> join_done{false};
    std::thread joiner([&]() {
        ex::sync_wait(scope.join());
        join_done.store(true, std::memory_order_release);
    });

    // The held operation is blocked; join() must not complete
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    HPX_TEST(!join_done.load(std::memory_order_acquire));

    // Release the held operation
    release.release();
    joiner.join();

    HPX_TEST(join_done.load(std::memory_order_acquire));
    HPX_TEST_EQ(completed.load(std::memory_order_acquire), n);
}

// spawn_future on a closed scope completes via set_stopped
void test_spawn_future_closed_scope()
{
    ex::simple_counting_scope scope;
    scope.close();

    auto fut = ex::spawn_future(ex::schedule(ex::thread_pool_scheduler{}) |
            ex::then([]() { return 99; }),
        scope.get_token());

    auto result = ex::sync_wait(std::move(fut) | ex::stopped_as_optional());
    HPX_TEST(result.has_value());
    auto [opt_val] = std::move(*result);
    HPX_TEST(!opt_val.has_value());

    ex::sync_wait(scope.join());
}

// counting_scope with request_stop: stop is delivered to HPX work.
// With a real thread pool, __stop_when may race the inner sender and
// complete with set_stopped before the work observes the token. Both
// outcomes (stopped, or value=true) confirm stop delivery.
void test_counting_scope_stop_with_scheduler()
{
    ex::counting_scope scope;
    scope.request_stop();

    auto fut = ex::spawn_future(
        ex::schedule(ex::thread_pool_scheduler{}) | ex::let_value([]() {
            return stdexec::read_env(stdexec::get_stop_token) |
                ex::then([](auto stoken) { return stoken.stop_requested(); });
        }),
        scope.get_token());

    // stopped_as_optional: set_stopped -> nullopt, set_value -> optional
    auto result = ex::sync_wait(std::move(fut) | ex::stopped_as_optional());
    HPX_TEST(result.has_value());
    auto [opt_val] = std::move(*result);

    // Either the sender was stopped (nullopt) or it observed the stop
    // token (true). Both confirm stop delivery.
    if (opt_val.has_value())
    {
        HPX_TEST(*opt_val);
    }

    scope.close();
    ex::sync_wait(scope.join());
}

int hpx_main(int, char*[])
{
    test_spawn_future_value();
    test_spawn_future_error();
    test_spawn_with_scheduler();
    test_join_blocks_for_async_work();
    test_spawn_future_closed_scope();
    test_counting_scope_stop_with_scheduler();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");
    return hpx::util::report_errors();
}

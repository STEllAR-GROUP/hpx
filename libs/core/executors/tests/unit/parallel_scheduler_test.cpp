// Copyright (c) 2025 Sai Charan Arvapally
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution.hpp>
<<<<<<< HEAD
#include <hpx/executors/thread_pool_scheduler.hpp>
=======
#include <hpx/executors/experimental/parallel_scheduler.hpp>
#include <hpx/executors/thread_pool_scheduler_bulk.hpp>
>>>>>>> 778b6d9505 (added bulk_unchunked functionality to thread_pool_scheduler_bulk)
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/synchronization/stop_token.hpp>
#include <hpx/thread.hpp>
#include <exception>
#include <future>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef HPX_HAVE_STDEXEC
#include <hpx/execution_base/stdexec_forward.hpp>
#endif

namespace ex = hpx::execution::experimental;

#ifdef HPX_HAVE_STDEXEC
// Forward declaration
struct test_receiver;

// Enable test_receiver as a valid stdexec receiver
namespace stdexec {
    template <>
    inline constexpr bool enable_receiver<::test_receiver> = true;

    // Define completion signatures for test_receiver
    template <>
    struct completion_signatures<::test_receiver>
    {
        using type = ex::completion_signatures<ex::set_value_t(),
            ex::set_error_t(std::exception_ptr), ex::set_stopped_t()>;
    };
}    // namespace stdexec

struct test_receiver
{
    struct state
    {
        bool completed = false;
        bool error_called = false;
        bool stopped_called = false;
        std::vector<std::thread::id>
            task_thread_ids;    // Track thread IDs for bulk tasks
        std::vector<uint32_t>
            executed_indices;    // Track executed task indices
    };

    std::shared_ptr<state> state_ = std::make_shared<state>();
    ex::inplace_stop_token stop_token;
    std::shared_ptr<std::promise<void>> done_promise =
        std::make_shared<std::promise<void>>();

    test_receiver() = default;
    test_receiver(test_receiver&&) = default;
    test_receiver& operator=(test_receiver&&) = default;
    test_receiver(const test_receiver&) = delete;
    test_receiver& operator=(const test_receiver&) = delete;

    std::future<void> get_future()
    {
        return done_promise->get_future();
    }

    void set_value() && noexcept
    {
        std::cout << "set_value called" << std::endl;
        state_->completed = true;
        done_promise->set_value();
    }

    void set_error([[maybe_unused]] std::exception_ptr ep) && noexcept
    {
        std::cout << "set_error called" << std::endl;
        state_->error_called = true;
        done_promise->set_value();
    }

    void set_stopped() && noexcept
    {
        std::cout << "set_stopped called" << std::endl;
        state_->stopped_called = true;
        done_promise->set_value();
    }
<<<<<<< HEAD
=======
#else
    void set_value() noexcept
    {
        std::cerr << "set_value not supported without stdexec" << std::endl;
    }

    void set_error([[maybe_unused]] std::exception_ptr) noexcept
    {
        std::cerr << "set_error not supported without stdexec" << std::endl;
    }

    void set_stopped() noexcept
    {
        std::cerr << "set_stopped not supported without stdexec" << std::endl;
    }
#endif
>>>>>>> 778b6d9505 (added bulk_unchunked functionality to thread_pool_scheduler_bulk)

    struct env
    {
        ex::inplace_stop_token token;

        friend auto tag_invoke(ex::get_stop_token_t, env const& e) noexcept
        {
            return e.token;
        }
    };

    env get_env() const noexcept
    {
        return {stop_token};
    }
};
#endif

int hpx_main(hpx::program_options::variables_map&)
{
    using scheduler_t = ex::thread_pool_policy_scheduler<hpx::launch>;
    scheduler_t sched(hpx::launch::async);

    // Test scheduler properties
    std::cout << "\n=== Scheduler Properties ===\n";
    {
        // Scheduler type
        std::cout << "Testing scheduler type\n";
        HPX_TEST((stdexec::scheduler<scheduler_t>) );

        // Destructible
        std::cout << "Testing destructibility\n";
        HPX_TEST(std::is_destructible_v<scheduler_t>);

        // Copyable and movable
        std::cout << "Testing copy/move constructibility\n";
        HPX_TEST(std::is_copy_constructible_v<scheduler_t>);
        HPX_TEST(std::is_move_constructible_v<scheduler_t>);

        // Copied scheduler equality
        std::cout << "Testing scheduler equality\n";
        auto sched2 = sched;
        HPX_TEST(sched == sched2);
    }

    // Test sender creation
    std::cout << "\n=== Sender Creation ===\n";
    {
#ifdef HPX_HAVE_STDEXEC
        auto snd = ex::schedule(sched);
        using sender_t = decltype(snd);
        std::cout << "Testing sender type\n";
        HPX_TEST(ex::sender<sender_t>);
        // Verify completion signatures
        std::cout << "Testing sender completion signatures\n";
        using expected_signatures = ex::completion_signatures<ex::set_value_t(),
            ex::set_error_t(std::exception_ptr), ex::set_stopped_t()>;
        HPX_TEST((std::is_same_v<
            ex::completion_signatures_of_t<sender_t, ex::empty_env>,
            expected_signatures>) );
#else
        std::cerr << "stdexec not enabled, skipping sender creation test\n";
#endif
    }

    // Test trivial schedule task
    std::cout << "\n=== Trivial Schedule Task ===\n";
    {
#ifdef HPX_HAVE_STDEXEC
        std::cout << "Scheduling trivial task\n";
        ex::sync_wait(ex::schedule(sched));
        std::cout << "Trivial task completed\n";
#else
        std::cerr
            << "stdexec not enabled, skipping trivial schedule task test\n";
#endif
    }

    // Test simple schedule task
    std::cout << "\n=== Simple Schedule Task ===\n";
    {
#ifdef HPX_HAVE_STDEXEC
        std::thread::id this_id = std::this_thread::get_id();
        std::thread::id pool_id{};
        std::cout << "Before scheduling simple task\n";
        auto snd = ex::then(
            ex::schedule(sched), [&] { pool_id = std::this_thread::get_id(); });
        std::cout << "Task scheduled, waiting for completion...\n";
        ex::sync_wait(std::move(snd));
        std::cout << "Simple task completed: pool_id != default: "
                  << (pool_id != std::thread::id{}) << "\n";
        std::cout << "Simple task completed: this_id != pool_id: "
                  << (this_id != pool_id) << "\n";
        HPX_TEST(pool_id != std::thread::id{});
        HPX_TEST(this_id != pool_id);
#else
        std::cerr
            << "stdexec not enabled, skipping simple schedule task test\n";
#endif
    }

    // Test forward progress guarantee
    std::cout << "\n=== Forward Progress Guarantee ===\n";
    {
        auto guarantee = ex::get_forward_progress_guarantee(sched);
        std::cout << "Forward progress guarantee: "
                  << static_cast<int>(guarantee) << "\n";
        std::cout << "Expected parallel: "
                  << static_cast<int>(ex::forward_progress_guarantee::parallel)
                  << ", concurrent: "
                  << static_cast<int>(
                         ex::forward_progress_guarantee::concurrent)
                  << ", weakly_parallel: "
                  << static_cast<int>(
                         ex::forward_progress_guarantee::weakly_parallel)
                  << "\n";
        HPX_TEST(guarantee == ex::forward_progress_guarantee::parallel);
    }

    // Test completion scheduler
    std::cout << "\n=== Completion Scheduler ===\n";
    {
#ifdef HPX_HAVE_STDEXEC
<<<<<<< HEAD
        std::cout << "Testing completion scheduler\n";
        bool is_same = ex::get_completion_scheduler<ex::set_value_t>(
                           ex::get_env(ex::schedule(sched))) == sched;
        std::cout << "Completion scheduler matches: " << is_same << "\n";
        HPX_TEST(is_same);
=======
    // Test single task (schedule)
    std::cout << "\n=== Single Task ===" << std::endl;
    {
        test_receiver recv;
        auto state = recv.state_;
        auto future = recv.get_future();
        auto sender = ex::schedule(sched);
        {
            auto op = ex::connect(std::move(sender), std::move(recv));
            std::cout << "Calling start() for single task" << std::endl;
            ex::start(op);
        }
        std::cout << "Waiting for single task to complete..." << std::endl;
        future.get();
        std::cout << "Single task completed: "
                  << (state->completed ? "true" : "false") << std::endl;
        HPX_TEST(state->completed);
    }

    // Test single task with cancellation
    std::cout << "\n=== Single Task With Cancellation ===" << std::endl;
    {
        test_receiver recv;
        auto state = recv.state_;
        auto future = recv.get_future();
        stdexec::inplace_stop_source stop_src;
        recv.stop_token = stop_src.get_token();
        auto sender = ex::schedule(sched);
        {
            auto op = ex::connect(std::move(sender), std::move(recv));
            std::cout << "Requesting stop before start..." << std::endl;
            stop_src.request_stop();
            std::cout << "Calling start() for single task with cancellation"
                      << std::endl;
            ex::start(op);
        }
        std::cout << "Waiting for single task with cancellation to complete..."
                  << std::endl;
        future.get();
        std::cout << "Single task with cancellation: stopped_called = "
                  << state->stopped_called
                  << ", completed = " << state->completed << std::endl;
        HPX_TEST(state->stopped_called && !state->completed);
    }

    // Test single task with exception
    std::cout << "\n=== Single Task With Exception ===" << std::endl;
    {
        test_receiver recv;
        auto state = recv.state_;
        auto future = recv.get_future();
        auto sender = ex::then(ex::schedule(sched), []() {
            std::cout << "Executing then functor" << std::endl;
            throw std::runtime_error("Test exception");
        });
        {
            auto op = ex::connect(std::move(sender), std::move(recv));
            std::cout << "Calling start() for single task with exception"
                      << std::endl;
            ex::start(op);
        }
        std::cout << "Waiting for single task with exception to complete..."
                  << std::endl;
        future.get();
        std::cout << "Single task with exception: error_called = "
                  << state->error_called << std::endl;
        HPX_TEST(state->error_called);
    }

    // Test bulk_unchunked
    std::cout << "\n=== Bulk Unchunked Task ===" << std::endl;
    {
        constexpr uint32_t num_tasks = 16;
        test_receiver recv;
        auto state = recv.state_;
        state->task_thread_ids.resize(num_tasks);
        state->executed_indices.resize(num_tasks, 0);
        auto future = recv.get_future();
        std::thread::id this_id = std::this_thread::get_id();

        auto sender = ex::schedule(sched) |
            stdexec::bulk(stdexec::par, num_tasks, [&](uint32_t idx) {
                state->task_thread_ids[idx] = std::this_thread::get_id();
                state->executed_indices[idx] = idx + 1;    // Mark as executed
                std::cout << "Bulk task " << idx << " on thread "
                          << state->task_thread_ids[idx] << std::endl;
            });

        {
            auto op = ex::connect(std::move(sender), std::move(recv));
            std::cout << "Calling start() for bulk_unchunked task" << std::endl;
            ex::start(op);
        }
        std::cout << "Waiting for bulk_unchunked task to complete..."
                  << std::endl;
        future.get();

        std::cout << "Bulk unchunked task completed: completed = "
                  << state->completed << std::endl;
        HPX_TEST(state->completed);
        for (uint32_t i = 0; i < num_tasks; ++i)
        {
            HPX_TEST(state->task_thread_ids[i] !=
                std::thread::id{});    // Non-empty thread ID
            HPX_TEST(state->task_thread_ids[i] !=
                std::thread::id{});    // Only ensure non-empty
            HPX_TEST(state->executed_indices[i] == i + 1);    // Task i executed
        }
    }

>>>>>>> 778b6d9505 (added bulk_unchunked functionality to thread_pool_scheduler_bulk)
#else
        std::cerr
            << "stdexec not enabled, skipping completion scheduler test\n";
#endif
    }

    // Test stop token before starting work
    std::cout << "\n=== Stop Token Before Starting Work ===\n";
    {
#ifdef HPX_HAVE_STDEXEC
        test_receiver recv;
        auto state = recv.state_;
        auto future = recv.get_future();
        ex::inplace_stop_source stop_src;
        recv.stop_token = stop_src.get_token();
        bool called = false;
        std::cout << "Before scheduling stop token task\n";
        auto snd = ex::then(ex::schedule(sched), [&called] { called = true; });
        std::cout << "Requesting stop before start...\n";
        stop_src.request_stop();
        std::cout << "After stop request, connecting task\n";
        auto op = ex::connect(std::move(snd), std::move(recv));
        std::cout << "Starting task\n";
        ex::start(op);
        std::cout << "Waiting for completion...\n";
        future.get();
        std::cout << "Stop token test: stopped_called = "
                  << state->stopped_called << ", called = " << called << "\n";
        HPX_TEST(state->stopped_called);
        HPX_TEST(!called);
#else
        std::cerr << "stdexec not enabled, skipping stop token test\n";
#endif
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::cout << "main() started\n";
    hpx::init_params init_args;
    std::cout << "Calling hpx::init\n";
    int result = hpx::init(hpx_main, argc, argv, init_args);
    std::cout << "hpx::init returned: " << result << "\n";
    return hpx::util::report_errors();
}

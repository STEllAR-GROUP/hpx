// Copyright (c) 2025 Sai Charan Arvapally
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution.hpp>
#include <hpx/executors/experimental/parallel_scheduler.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/synchronization/stop_token.hpp>
#include <hpx/thread.hpp>
#include <exception>
#include <future>
#include <iostream>
#include <memory>

namespace ex = hpx::execution::experimental;

struct test_receiver
{
    struct state
    {
        bool completed = false;
        bool error_called = false;
        bool stopped_called = false;
    };

    std::shared_ptr<state> state_ = std::make_shared<state>();
    hpx::stop_token stop_token;
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

    friend void tag_invoke(ex::set_value_t, test_receiver&& r) noexcept
    {
        std::cout << "set_value called" << std::endl;
        r.state_->completed = true;
        r.done_promise->set_value();
    }

    friend void tag_invoke(
        ex::set_error_t, test_receiver&& r, std::exception_ptr ep) noexcept
    {
        (void) ep;
        std::cout << "set_error called" << std::endl;
        r.state_->error_called = true;
        r.done_promise->set_value();
    }

    friend void tag_invoke(ex::set_stopped_t, test_receiver&& r) noexcept
    {
        std::cout << "set_stopped called" << std::endl;
        r.state_->stopped_called = true;
        r.done_promise->set_value();
    }

    struct env
    {
        hpx::stop_token token;

        friend auto tag_invoke(hpx::execution::experimental::get_stop_token_t,
            env const& e) noexcept
        {
            return e.token;
        }
    };

    friend env tag_invoke(hpx::execution::experimental::get_env_t,
        test_receiver const& r) noexcept
    {
        return {r.stop_token};
    }
};

int hpx_main(hpx::program_options::variables_map&)
{
    std::cout << "hpx_main started" << std::endl;
    ex::parallel_scheduler sched = ex::get_parallel_scheduler();

    // Test forward progress guarantee
    std::cout << "\n=== Forward Progress Guarantee ===" << std::endl;
    {
        auto guarantee = ex::get_forward_progress_guarantee(sched);
        std::cout << "Forward progress guarantee: "
                  << static_cast<int>(guarantee) << std::endl;
        HPX_TEST(guarantee == ex::forward_progress_guarantee::parallel);
    }

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
        hpx::stop_source stop_src;
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

    std::cout << "Calling hpx::finalize()" << std::endl;
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::cout << "main() started" << std::endl;
    hpx::init_params init_args;
    std::cout << "Calling hpx::init" << std::endl;
    int result = hpx::init(hpx_main, argc, argv, init_args);
    std::cout << "hpx::init returned: " << result << std::endl;
    return hpx::util::report_errors();
}

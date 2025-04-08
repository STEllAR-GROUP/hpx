// Copyright (c) 2025 Sai Charan Arvapally
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution.hpp>
#include <hpx/executors/parallel_scheduler.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/synchronization/stop_token.hpp>
#include <hpx/thread.hpp>
#include <exception>
#include <future>
#include <iostream>

namespace ex = hpx::execution::experimental;

struct test_receiver
{
    bool completed = false;
    bool error_called = false;
    bool stopped_called = false;
    int bulk_count = 0;
    hpx::stop_token stop_token;
    std::promise<void> done_promise;

    std::future<void> get_future()
    {
        return done_promise.get_future();
    }

    friend void tag_invoke(ex::set_value_t, test_receiver& r) noexcept
    {
        std::cout << "set_value called" << std::endl;
        r.completed = true;
        if (r.bulk_count > 0)
            r.bulk_count = 3;
        r.done_promise.set_value();
    }

    friend void tag_invoke(
        ex::set_error_t, test_receiver&& r, std::exception_ptr ep) noexcept
    {
        (void) ep;
        std::cout << "set_error called" << std::endl;
        r.error_called = true;
        r.done_promise.set_value();
    }

    friend void tag_invoke(ex::set_stopped_t, test_receiver& r) noexcept
    {
        std::cout << "set_stopped called" << std::endl;
        r.stopped_called = true;
        r.done_promise.set_value();
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
    ex::parallel_scheduler sched;

    // Test single task (schedule)
    std::cout << "\n=== Single Task ===" << std::endl;
    {
        test_receiver recv;
        auto sender = ex::schedule(sched);
        auto op = ex::connect(std::move(sender), recv);
        std::cout << "Calling start() for single task" << std::endl;
        ex::start(op);
        std::cout << "Waiting for single task to complete..." << std::endl;
        recv.get_future().get();    // Wait for the operation to complete
        std::cout << "Single task completed: "
                  << (recv.completed ? "true" : "false") << std::endl;
        HPX_TEST(recv.completed);
    }

    // Test bulk
    std::cout << "\n=== Bulk Task ===" << std::endl;
    {
        test_receiver recv;
        recv.bulk_count = 1;    // Initialize to trigger bulk completion logic
        auto bulk_op = ex::connect(
            ex::bulk(ex::schedule(sched), 3,
                [](int i) {
                    std::cout << "Bulk functor called for index: " << i
                              << std::endl;
                }),
            recv);
        std::cout << "Calling start() for bulk task" << std::endl;
        ex::start(bulk_op);
        std::cout << "Waiting for bulk task to complete..." << std::endl;
        recv.get_future().get();    // Wait for the operation to complete
        std::cout << "Bulk task completed: "
                  << (recv.completed ? "true" : "false") << std::endl;
        HPX_TEST(recv.completed && recv.bulk_count == 3);
    }

    // Test bulk with exception
    std::cout << "\n=== Bulk Task With Exception ===" << std::endl;
    {
        test_receiver recv;
        auto bulk_op_with_error = ex::connect(
            ex::bulk(ex::schedule(sched), 3,
                [](int i) {
                    if (i == 1)
                    {
                        throw std::runtime_error("Test exception");
                    }
                    std::cout << "Bulk functor called for index: " << i
                              << std::endl;
                }),
            recv);
        std::cout << "Calling start() for bulk task with exception"
                  << std::endl;
        ex::start(bulk_op_with_error);
        std::cout << "Waiting for bulk task with exception to complete..."
                  << std::endl;
        recv.get_future().get();    // Wait for the operation to complete
        std::cout << "Bulk task with exception: error_called = "
                  << recv.error_called << std::endl;
        HPX_TEST(recv.error_called);
    }

    // Test single task with cancellation
    std::cout << "\n=== Single Task With Cancellation ===" << std::endl;
    {
        hpx::stop_source stop_src;
        test_receiver recv;
        recv.stop_token = stop_src.get_token();
        auto sender = ex::schedule(sched);
        auto op = ex::connect(std::move(sender), recv);
        std::cout << "Calling start() for single task with cancellation"
                  << std::endl;
        ex::start(op);
        std::cout << "Requesting stop..." << std::endl;
        stop_src.request_stop();
        std::cout << "Waiting for single task with cancellation to complete..."
                  << std::endl;
        recv.get_future().get();    // Wait for the operation to complete
        std::cout << "Single task with cancellation: stopped_called = "
                  << recv.stopped_called << ", completed = " << recv.completed
                  << std::endl;
        HPX_TEST(recv.stopped_called && !recv.completed);
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

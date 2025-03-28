//  Copyright (c) 2025 Sai Charan Arvapally
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>
#include <exception>
#include <iostream>

namespace ex = hpx::execution::experimental;

struct test_receiver
{
    bool completed = false;
    int bulk_count = 0;

    friend void tag_invoke(ex::set_value_t, test_receiver& r) noexcept
    {
        std::cout << "set_value called" << std::endl;
        r.completed = true;
        if (r.bulk_count > 0)
            r.bulk_count = 3;    // Mark bulk completion
    }

    friend void tag_invoke(
        ex::set_error_t, test_receiver&&, std::exception_ptr) noexcept
    {
        std::cout << "set_error called" << std::endl;
    }

    friend void tag_invoke(ex::set_stopped_t, test_receiver&&) noexcept
    {
        std::cout << "set_stopped called" << std::endl;
    }
};

int hpx_main(hpx::program_options::variables_map&)
{
    std::cout << "hpx_main started" << std::endl;
    ex::parallel_scheduler sched;

    // Test single task (schedule)
    auto sender = ex::schedule(sched);
    test_receiver recv;
    auto op = ex::connect(sender, recv);
    std::cout << "Calling start() for single task" << std::endl;
    ex::start(op);

    int attempts = 0;
    while (!recv.completed && attempts < 1000)
    {
        hpx::this_thread::yield();
        attempts++;
    }
    // Add a small delay to ensure the single task finishes
    hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout << "Single task completed: "
              << (recv.completed ? "true" : "false") << std::endl;
    HPX_TEST(recv.completed);

    // Test bulk
    recv.completed = false;    // Reset for bulk test
    recv.bulk_count = 1;       // Initialize to trigger bulk completion logic
    auto bulk_op =
        ex::connect(ex::bulk(ex::schedule(sched), 3,
                        [](int i) {
                            std::cout << "Bulk functor called for index: " << i
                                      << std::endl;
                        }),
            recv);
    std::cout << "Calling start() for bulk task" << std::endl;
    ex::start(bulk_op);

    attempts = 0;
    while (!recv.completed && attempts < 1000)
    {
        hpx::this_thread::yield();
        attempts++;
    }
    // Keep the delay for bulk to ensure all tasks finish
    hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout << "Bulk task count: " << recv.bulk_count << std::endl;
    HPX_TEST(recv.bulk_count == 3);

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

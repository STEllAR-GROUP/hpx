//  Copyright (c) 2025 Arpit Khandelwal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example demonstrates the use of hpx::local::termination_detection()
// to wait for all asynchronous work to complete before shutting down the runtime.

#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/runtime_local/termination_detection.hpp>
#include <hpx/stop_token.hpp>

#include <iostream>
#include <vector>

//[termination_detection_example
int hpx_main()
{
    // Example 1: Basic usage - Wait for all local threads to complete
    {
        std::cout << "Example 1: Basic usage" << std::endl;
        std::cout << "Starting async work..." << std::endl;

        // Launch 100 asynchronous tasks
        for (int i = 0; i < 100; ++i)
        {
            hpx::post([i]() {
                // Simulate some work
                hpx::this_thread::sleep_for(std::chrono::milliseconds(10));
            });
        }

        std::cout << "Waiting for queries to pass..." << std::endl;
        hpx::local::termination_detection();
        std::cout << "Basic wait completed." << std::endl;
    }

    // Example 2: Wait with timeout
    {
        std::cout << "\nExample 2: Wait with timeout" << std::endl;
        hpx::post([] {
            hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
        });

        bool completed = hpx::local::termination_detection(
            hpx::chrono::steady_duration(std::chrono::milliseconds(500)));

        if (completed)
            std::cout << "Completed within timeout." << std::endl;
        else
            std::cout << "Timed out." << std::endl;
    }

    // Example 3: Wait with deadline
    {
        std::cout << "\nExample 3: Wait with deadline" << std::endl;
        hpx::post([] {
            hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
        });

        auto deadline =
            hpx::chrono::steady_clock::now() + std::chrono::milliseconds(500);

        // Note: implicit conversion from steady_clock::time_point to steady_time_point
        bool completed = hpx::local::termination_detection(deadline);

        if (completed)
            std::cout << "Completed before deadline." << std::endl;
        else
            std::cout << "Deadline reached." << std::endl;
    }

    // Example 4: Wait with cancellation support
    {
        std::cout << "\nExample 4: Wait with cancellation" << std::endl;
        hpx::stop_source stop_src;

        // Launch a long running task
        hpx::post([] { hpx::this_thread::sleep_for(std::chrono::seconds(2)); });

        // Trigger cancellation after 100ms
        hpx::thread canceller([&stop_src]() {
            hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::cout << "Requesting stop..." << std::endl;
            stop_src.request_stop();
        });

        bool completed =
            hpx::local::termination_detection(stop_src.get_token());

        if (!completed)
            std::cout << "Wait cancelled or timed out." << std::endl;
        else
            std::cout << "Wait completed successfully." << std::endl;

        canceller.join();
    }

    std::cout << "\nAll work completed. Shutting down." << std::endl;

    return hpx::finalize();
}
//]

int main(int argc, char* argv[])
{
    return hpx::init(hpx_main, argc, argv);
}

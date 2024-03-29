//  Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
//                     LLC (NTESS).
//  Copyright (c) 2019 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/chrono.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/resiliency.hpp>

#include <atomic>
#include <cstdint>
#include <ctime>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

constexpr int num_iterations = 1000;

std::random_device rd;
std::mt19937 gen(rd());

struct vogon_exception : std::exception
{
};

std::atomic<int> counter(0);

bool validate(int result)
{
    return result == 42;
}

int universal_ans(std::uint64_t delay_ns, double error)
{
    std::exponential_distribution<> dist(error);

    if (delay_ns == 0)
        return 42;

    double num = dist(gen);
    bool error_flag = false;

    // Probability of error occurrence is proportional to exp(-error_rate)
    if (num > 1.0)
    {
        error_flag = true;
        ++counter;
    }

    std::uint64_t start = hpx::chrono::high_resolution_clock::now();

    while (true)
    {
        // Check if we've reached the specified delay.
        if ((hpx::chrono::high_resolution_clock::now() - start) >= delay_ns)
        {
            // Re-run the thread if the thread was meant to re-run
            if (error_flag)
                throw vogon_exception();
            // No error has to occur with this thread, simply break the loop after
            // execution is done for the desired time
            else
                break;
        }
    }

    return 42;
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    std::uint64_t n = vm["n-value"].as<std::uint64_t>();
    double error = vm["error-rate"].as<double>();
    std::uint64_t delay = vm["exec-time"].as<std::uint64_t>();

    {
        std::cout << "Starting async replay" << std::endl;

        std::vector<hpx::future<int>> vect;
        vect.reserve(num_iterations);

        hpx::chrono::high_resolution_timer t;

        for (int i = 0; i < num_iterations; ++i)
        {
            hpx::future<int> f = hpx::resiliency::experimental::async_replay(
                n, &universal_ans, delay * 1000, error);
            vect.push_back(std::move(f));
        }

        try
        {
            for (int i = 0; i < num_iterations; ++i)
            {
                vect[i].get();
            }
        }
        catch (vogon_exception const&)
        {
            std::cout << "Number of repeat launches were not enough to get "
                         "past the injected error levels"
                      << std::endl;
        }
        catch (hpx::resiliency::experimental::abort_replay_exception const&)
        {
            std::cout << "Number of repeat launches were not enough to get "
                         "past the injected error levels"
                      << std::endl;
        }

        double elapsed = t.elapsed();
        hpx::util::format_to(std::cout,
            "Async replay execution time = {1}\n"
            "Number of failed threads = {2}\n",
            elapsed, counter);
    }

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    using hpx::program_options::options_description;
    using hpx::program_options::value;

    // Configure application-specific options
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("n-value",
        value<std::uint64_t>()->default_value(10),
        "Number of repeat launches for async replay");

    desc_commandline.add_options()("error-rate",
        value<double>()->default_value(2),
        "Average rate at which error is likely to occur");

    desc_commandline.add_options()("exec-time",
        value<std::uint64_t>()->default_value(1000),
        "Time in us taken by a thread to execute before it terminates");

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return hpx::local::init(hpx_main, argc, argv, init_args);
}

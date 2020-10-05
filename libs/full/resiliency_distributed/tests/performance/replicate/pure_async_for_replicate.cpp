//  Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
//                     LLC (NTESS).
//  Copyright (c) 2019 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/future.hpp>
#include <hpx/modules/resiliency.hpp>
#include <hpx/modules/timing.hpp>

#include <atomic>
#include <cstdint>
#include <ctime>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

std::random_device rd;
std::mt19937 gen(rd());

struct vogon_exception : std::exception
{
};

bool validate(std::size_t result)
{
    return result == 42;
}

std::size_t vote(std::vector<std::size_t>&& vect)
{
    return std::move(vect.at(0));
}

std::size_t universal_ans(std::size_t delay_ns, std::size_t error)
{
    std::uniform_int_distribution<std::size_t> dist(1, 100);

    std::size_t start = hpx::util::high_resolution_clock::now();

    while (true)
    {
        // Check if we've reached the specified delay.
        if ((hpx::util::high_resolution_clock::now() - start) >=
            (delay_ns * 1e3))
        {
            // Re-run the thread if the thread was meant to re-run
            if (dist(gen) < error)
                throw vogon_exception();

            break;
        }
    }

    return 42;
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    std::size_t n = vm["n-value"].as<std::size_t>();
    std::size_t error = vm["error"].as<std::size_t>();
    std::size_t delay = vm["size"].as<std::size_t>();
    std::size_t num_iterations = vm["num-iterations"].as<std::size_t>();

    {
        std::cout << "Starting async" << std::endl;

        std::vector<hpx::future<std::size_t>> vect;
        vect.reserve(num_iterations);

        hpx::util::high_resolution_timer t;

        for (std::size_t i = 0; i < num_iterations; ++i)
        {
            hpx::future<std::size_t> f = hpx::async(&universal_ans, delay, 0);
            vect.push_back(std::move(f));
        }

        hpx::wait_all(vect);

        double elapsed = t.elapsed();
        hpx::util::format_to(
            std::cout, "Pure Async execution time = {1}\n", elapsed);
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using hpx::program_options::options_description;
    using hpx::program_options::value;

    // Configure application-specific options
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("n-value",
        value<std::size_t>()->default_value(3),
        "Number of asynchronous launches for async replicate");

    desc_commandline.add_options()("error",
        value<std::size_t>()->default_value(2),
        "Percentage error to inject in the code");

    desc_commandline.add_options()("size",
        value<std::size_t>()->default_value(1000),
        "Time in us taken by a thread to execute before it terminates");

    desc_commandline.add_options()("num-iterations",
        value<std::size_t>()->default_value(1e6), "Number of tasks");

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}

//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>
#include <hpx/local/chrono.hpp>
#include <hpx/local/execution.hpp>
#include <hpx/local/future.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/modules/testing.hpp>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

bool use_scoping = true;

///////////////////////////////////////////////////////////////////////////////
std::uint64_t fibonacci(std::uint64_t n)
{
    if (n < 2)
        return n;

    hpx::threads::thread_schedule_hint hint;
    hint.runs_as_child = use_scoping;

    auto exec = hpx::execution::experimental::with_hint(
        hpx::execution::parallel_executor{}, hint);

    hpx::future<std::uint64_t> n1 = hpx::async(exec, fibonacci, n - 1);
    std::uint64_t n2 = fibonacci(n - 2);

    // wait for the Futures to return their values
    return n1.get() + n2;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    // extract command line argument, i.e. fib(N)
    std::uint64_t n = vm["n-value"].as<std::uint64_t>();

    use_scoping = vm.count("non-scoped") == 0;

    {
        // Keep track of the time required to execute.
        hpx::chrono::high_resolution_timer t;

        std::uint64_t r = fibonacci(n);

        char const* fmt = "fibonacci({1}) == {2}\nelapsed time: {3} [s]\n";
        hpx::util::format_to(std::cout, fmt, n, r, t.elapsed());
    }

    return hpx::finalize();    // Handles HPX shutdown
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    hpx::program_options::options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // clang-format off
    desc_commandline.add_options()
        ("n-value",
            hpx::program_options::value<std::uint64_t>()->default_value(10),
            "n value for the Fibonacci function")
        ("non-scoped", "run created threads without scoping");
    // clang-format on

    // use LIFO scheduler
    hpx::init_params params;
    params.desc_cmdline = desc_commandline;

    // Initialize and run HPX
    hpx::init(argc, argv, params);

    return hpx::util::report_errors();
}

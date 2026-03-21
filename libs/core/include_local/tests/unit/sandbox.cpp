//  Copyright (c) 2020-2026 The STE||AR-Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// HPX Sandbox Laboratory — unit test
//
// Verifies hpx/experimental/sandbox.hpp by:
//   1. Introspecting the environment (describe_environment)
//   2. Running a comparative for_each benchmark (seq vs par)
//   3. Printing the full telemetry report
//
// See PR #7079 for design details and compliance constraints.

#include <hpx/config.hpp>

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/experimental/sandbox.hpp>
#include <hpx/init.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <vector>

int hpx_main()
{
    namespace sandbox = hpx::experimental::sandbox;

    // 1. Print environment report
    sandbox::describe_environment(std::cout);

    // 2. Prepare test data — large enough to show parallel benefit
    constexpr std::size_t N = 500'000;
    std::vector<double> data(N);
    std::iota(data.begin(), data.end(), 1.0);

    // 3. Define a non-trivial work function to prevent optimisation
    auto work = [](double& x) { x = std::sin(x) * std::cos(x) + std::sqrt(x); };

    // 4. Run comparative benchmark
    auto report = sandbox::benchmark(
        "for_each (sin*cos+sqrt on 500k doubles)",
        [&]() {
            hpx::for_each(hpx::execution::seq, data.begin(), data.end(), work);
        },
        [&]() {
            hpx::for_each(hpx::execution::par, data.begin(), data.end(), work);
        },
        5    // iterations
    );

    // 5. Print the benchmark report
    report.print(std::cout);

    // 6. Sanity checks (non-fatal)
    if (report.speedup < 0.0)
    {
        std::cerr << "ERROR: negative speedup\n";
        return 1;
    }

    std::cout << "\nSandbox test passed.\n";
    return hpx::local::finalize();
}

int main(int argc, char** argv)
{
    return hpx::local::init(hpx_main, argc, argv);
}

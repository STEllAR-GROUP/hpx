//  Copyright (c) 2026 Vansh Dobhal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Causal tracing smoke test - value fulfillment path.
//
// PURPOSE: Validate that set_value() emits a producer-side causal signal
//          that precedes the consumer continuation in Tracy.
//
// TRACY USAGE:
//   1. Start Tracy profiler GUI.
//   2. Run:  TRACY_NO_EXIT=1 ./bin/causal_chain_smoke --iterations=200
//   3. Connect Tracy to the process.
//   4. In the Tracy message log, filter by color 0x00FF00 (bright green).
//      You should see repeated:
//        "Future Fulfilled: 0x<addr>"  [bright green]
//      immediately followed by consumer-side message:
//        "Continuation Run: 0x<addr>" [white]

#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/tracing.hpp>
#include <hpx/thread.hpp>

#include <chrono>
#include <cstddef>
#include <iostream>

int hpx_main(hpx::program_options::variables_map& vm)
{
    std::size_t const iterations = vm["iterations"].as<std::size_t>();

    std::cout << "causal_chain_smoke: starting (" << iterations
              << " iterations)\n"
              << std::flush;

    for (std::size_t i = 0; i < iterations; ++i)
    {
        hpx::tracing::frame_mark("smoke_iteration");

        hpx::promise<int> p;
        hpx::future<int> f = p.get_future();

        // Attach continuation *before* set_value so it goes through the
        // on_completed_ callback path (not the already-ready shortcut).
        hpx::future<int> result =
            f.then([](hpx::future<int> fut) { return fut.get() * 2; });

        // --- Producer side ------------------------------------------------
        // set_value() emits:
        //   [green] "Future Fulfilled: 0x<future_data*>"
        // then wakes the consumer, which fires:
        //   [white] "Continuation Run: 0x<future_data*>"
        p.set_value(21);

        int const value = result.get();
        HPX_TEST_EQ(value, 42);

        // Brief pause so each iteration is clearly visible in Tracy timeline
        hpx::this_thread::sleep_for(std::chrono::milliseconds(150));

        if (iterations >= 20 && i % (iterations / 10) == 0)
        {
            std::cout << "  iteration " << i << "/" << iterations << " OK\n"
                      << std::flush;
        }
    }

    std::cout << "causal_chain_smoke completed (" << iterations
              << " iterations)\n";

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    hpx::program_options::options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");
    desc_commandline.add_options()("iterations",
        hpx::program_options::value<std::size_t>()->default_value(5),
        "Number of test iterations to execute (default: 5 for fast CI)");

    hpx::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = {"hpx.os_threads=2"};

    HPX_TEST_EQ(hpx::local::init(hpx_main, argc, argv, init_args), 0);
    return hpx::util::report_errors();
}

//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>
#include <hpx/local/barrier.hpp>
#include <hpx/local/functional.hpp>
#include <hpx/local/thread.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using hpx::program_options::options_description;
using hpx::program_options::value;
using hpx::program_options::variables_map;

using hpx::threads::register_work;

using hpx::finalize;
using hpx::init;

using hpx::util::report_errors;

///////////////////////////////////////////////////////////////////////////////
void suspend_test(std::shared_ptr<hpx::barrier<>> b, std::size_t iterations)
{
    for (std::size_t i = 0; i < iterations; ++i)
    {
        // Enter the 'pending' state and get rescheduled.
        hpx::this_thread::suspend(
            hpx::threads::thread_schedule_state::pending, "suspend_test");
    }

    // Wait for all hpx-threads to enter the barrier.
    b->arrive_and_wait();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    std::size_t pxthreads = 0;

    if (vm.count("pxthreads"))
        pxthreads = vm["pxthreads"].as<std::size_t>();

    std::size_t iterations = 0;

    if (vm.count("iterations"))
        iterations = vm["iterations"].as<std::size_t>();

    {
        std::shared_ptr<hpx::barrier<>> b =
            std::make_shared<hpx::barrier<>>(pxthreads + 1);

        // Create the hpx-threads.
        for (std::size_t i = 0; i < pxthreads; ++i)
        {
            hpx::threads::thread_init_data data(
                hpx::threads::make_thread_function_nullary(
                    hpx::bind(&suspend_test, b, iterations)),
                "suspend_test");
            register_work(data);
        }

        b->arrive_and_wait();    // Wait for all hpx-threads to enter the barrier.
    }

    // Initiate shutdown of the runtime system.
    return finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("pxthreads,T",
        value<std::size_t>()->default_value(0x100),
        "the number of PX threads to invoke")("iterations",
        value<std::size_t>()->default_value(64),
        "the number of iterations to execute in each thread");

    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, init_args), 0,
        "HPX main exited with non-zero status");
    return report_errors();
}

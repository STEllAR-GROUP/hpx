//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/barrier.hpp>
#include <hpx/functional.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using hpx::program_options::options_description;
using hpx::program_options::value;
using hpx::program_options::variables_map;

using hpx::threads::make_thread_function_nullary;
using hpx::threads::register_work;
using hpx::threads::thread_init_data;

using hpx::barrier;

using hpx::local::finalize;
using hpx::local::init;

using hpx::util::report_errors;

///////////////////////////////////////////////////////////////////////////////
void local_barrier_test(
    std::shared_ptr<hpx::barrier<>> b, std::atomic<std::size_t>& c)
{
    ++c;
    // wait for all threads to enter the barrier
    b->arrive_and_wait();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    std::size_t threads = 0;

    if (vm.count("threads"))
        threads = vm["threads"].as<std::size_t>();

    std::size_t iterations = 0;

    if (vm.count("iterations"))
        iterations = vm["iterations"].as<std::size_t>();

    for (std::size_t i = 0; i < iterations; ++i)
    {
        // create a barrier waiting on 'count' threads
        std::shared_ptr<hpx::barrier<>> b =
            std::make_shared<hpx::barrier<>>(threads + 1);

        std::atomic<std::size_t> c(0);

        // create the threads which will wait on the barrier
        for (std::size_t i = 0; i < threads; ++i)
        {
            thread_init_data data(
                make_thread_function_nullary(
                    hpx::bind(&local_barrier_test, std::ref(b), std::ref(c))),
                "local_barrier_test");
            register_work(data);
        }

        // wait for all threads to enter the barrier
        b->arrive_and_wait();
        HPX_TEST_EQ(threads, c);
    }

    // initiate shutdown of the runtime system
    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("threads,T",
        value<std::size_t>()->default_value(64),
        "the number of PX threads to invoke")("iterations",
        value<std::size_t>()->default_value(64),
        "the number of times to repeat the test");

    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");
    return report_errors();
}

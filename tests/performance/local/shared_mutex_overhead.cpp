//  Copyright (c) 2026 Arpit Khandelwal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/timing.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <vector>

using hpx::program_options::options_description;
using hpx::program_options::value;
using hpx::program_options::variables_map;

using hpx::chrono::high_resolution_timer;

///////////////////////////////////////////////////////////////////////////////
hpx::shared_mutex mtx;
std::uint64_t num_iterations = 0;
std::size_t reader_threads = 0;

void reader_function()
{
    for (std::uint64_t i = 0; i < num_iterations; ++i)
    {
        std::shared_lock<hpx::shared_mutex> l(mtx);
        // Do some very light work
        hpx::util::detail::yield_k(10, "reader_work");
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    num_iterations = vm["iterations"].as<std::uint64_t>();
    reader_threads = hpx::get_num_worker_threads();

    std::cout << "Starting benchmark with " << reader_threads << " threads..."
              << std::endl;

    std::vector<hpx::future<void>> futures;
    futures.reserve(reader_threads);

    high_resolution_timer walltime;

    for (std::size_t i = 0; i < reader_threads; ++i)
    {
        futures.push_back(hpx::async(&reader_function));
    }

    hpx::wait_all(futures);

    double const duration = walltime.elapsed();

    std::cout << "Total time: " << duration << " seconds" << std::endl;
    std::cout << "Average time per reader thread: " << duration / reader_threads
              << " seconds" << std::endl;

    hpx::util::print_cdash_timing("SharedMutexOverhead", duration);

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()("iterations",
        value<std::uint64_t>()->default_value(100000),
        "number of iterations per thread");

    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return hpx::init(argc, argv, init_args);
}
#endif

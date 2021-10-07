//  Copyright (c) 2015 Martin Stumpf
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test demonstrates the issue described in #1422: hpx:threads=all
// allocates too many os threads

#include <hpx/local/init.hpp>
#include <hpx/local/runtime.hpp>
#include <hpx/local/thread.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

unsigned long num_cores = 0;

int hpx_main()
{
    std::size_t const os_threads = hpx::get_os_thread_count();

    std::cout << "Cores: " << num_cores << std::endl;
    std::cout << "OS Threads: " << os_threads << std::endl;

    HPX_TEST_EQ(num_cores, os_threads);

    return hpx::local::finalize();
}

int main(int argc, char** argv)
{
    // Get number of cores from OS
    num_cores = hpx::threads::hardware_concurrency();

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");
    return hpx::util::report_errors();
}

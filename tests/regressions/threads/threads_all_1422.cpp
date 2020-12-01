//  Copyright (c) 2015 Martin Stumpf
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test demonstrates the issue described in #1422: hpx:threads=all
// allocates too many os threads

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
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

    return hpx::finalize();
}

int main(int argc, char **argv)
{
    // Get number of cores from OS
    num_cores = hpx::threads::hardware_concurrency();

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, init_args), 0,
        "HPX main exited with non-zero status");
    return hpx::util::report_errors();
}

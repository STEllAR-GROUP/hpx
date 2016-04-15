//  Copyright (c) 2015 Martin Stumpf
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test demonstrates the issue described in #1422: hpx:threads=all
// allocates too many os threads

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <string>
#include <vector>

unsigned long num_cores = 0;

int hpx_main(int argc, char ** argv)
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
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=all");

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");
    return hpx::util::report_errors();
}

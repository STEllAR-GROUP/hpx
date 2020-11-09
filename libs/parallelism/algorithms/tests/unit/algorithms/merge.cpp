//  Copyright (c) 2017 Taeguk Kwon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "merge_tests.hpp"
#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
void merge_test()
{
    std::cout << "--- merge_test ---" << std::endl;
    test_merge<std::random_access_iterator_tag>();
    //test_merge<std::bidirectional_iterator_tag>();
    //test_merge<std::forward_iterator_tag>();
}

void merge_exception_test()
{
    std::cout << "--- merge_exception_test ---" << std::endl;
    test_merge_exception<std::random_access_iterator_tag>();
    //test_merge_exception<std::bidirectional_iterator_tag>();
    //test_merge_exception<std::forward_iterator_tag>();
}

void merge_bad_alloc_test()
{
    std::cout << "--- merge_bad_alloc_test ---" << std::endl;
    test_merge_bad_alloc<std::random_access_iterator_tag>();
    //test_merge_bad_alloc<std::bidirectional_iterator_tag>();
    //test_merge_bad_alloc<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = std::random_device{}();
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    merge_seed(seed);
    std::cout << "using seed: " << seed << std::endl;

    merge_test();
    merge_exception_test();
    merge_bad_alloc_test();

    std::cout << "Test Finish!" << std::endl;

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace hpx::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run");

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

//  Copyright (c) 2024 Tobias Wukovitsch
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>

#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "mismatch_binary_tests.hpp"

template <typename IteratorTag>
void mismatch_binary_sender_test()
{
    using namespace hpx::execution;

    test_mismatch_binary1_sender(hpx::launch::sync, seq(task), IteratorTag());
    test_mismatch_binary1_sender(hpx::launch::sync, unseq(task), IteratorTag());
    test_mismatch_binary1_sender(hpx::launch::async, par(task), IteratorTag());
    test_mismatch_binary1_sender(
        hpx::launch::async, par_unseq(task), IteratorTag());

    test_mismatch_binary2_sender(hpx::launch::sync, seq(task), IteratorTag());
    test_mismatch_binary2_sender(hpx::launch::sync, unseq(task), IteratorTag());
    test_mismatch_binary2_sender(hpx::launch::async, par(task), IteratorTag());
    test_mismatch_binary2_sender(
        hpx::launch::async, par_unseq(task), IteratorTag());
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    mismatch_binary_sender_test<std::forward_iterator_tag>();
    mismatch_binary_sender_test<std::random_access_iterator_tag>();

    return hpx::local::finalize();
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
    hpx::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

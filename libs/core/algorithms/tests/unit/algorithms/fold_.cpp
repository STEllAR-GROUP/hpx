//  Copyright (c) 2014-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "fold_tests.hpp"

template <typename IteratorTag>
void fold_left_test_dispatch_2(IteratorTag)
{
    fold_left_test1(IteratorTag());
    fold_left_test1(hpx::execution::seq, IteratorTag());
    fold_left_test1(hpx::execution::par, IteratorTag());
}

void fold_left_test_dispatch_1()
{
    fold_left_test_dispatch_2(std::random_access_iterator_tag());
    fold_left_test_dispatch_2(std::forward_iterator_tag());
}

void fold_left_test_dispatch()
{
    fold_left_test_dispatch_1();
}

template <typename IteratorTag>
void fold_right_test_dispatch_2(IteratorTag)
{
    fold_right_test1(IteratorTag());
    fold_right_test1(hpx::execution::seq, IteratorTag());
    fold_right_test1(hpx::execution::par, IteratorTag());
}

void fold_right_test_dispatch_1()
{
    fold_right_test_dispatch_2(std::random_access_iterator_tag());
    // fold_right_test_dispatch_2(std::forward_iterator_tag());
}

void fold_right_test_dispatch()
{
    fold_right_test_dispatch_1();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    fold_left_test_dispatch();
    fold_right_test_dispatch();

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

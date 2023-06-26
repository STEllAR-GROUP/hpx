//  Copyright (c) 2021 Srinivas Yadav
//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "find_tests.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_find_explicit_sender_direct()
{
    using namespace hpx::execution;

    test_find_explicit_sender_direct(hpx::launch::sync, seq, IteratorTag());
    test_find_explicit_sender_direct(hpx::launch::sync, unseq, IteratorTag());

    test_find_explicit_sender_direct(hpx::launch::async, par, IteratorTag());
    test_find_explicit_sender_direct(
        hpx::launch::async, par_unseq, IteratorTag());

    test_find_explicit_sender_direct_async(
        hpx::launch::sync, seq(task), IteratorTag());
    test_find_explicit_sender_direct_async(
        hpx::launch::sync, unseq(task), IteratorTag());
    test_find_explicit_sender_direct_async(
        hpx::launch::async, par(task), IteratorTag());
    test_find_explicit_sender_direct_async(
        hpx::launch::async, par_unseq(task), IteratorTag());
}

void find_test_explicit_sender_direct()
{
    test_find_explicit_sender_direct<std::random_access_iterator_tag>();
    test_find_explicit_sender_direct<std::forward_iterator_tag>();
}

template <typename IteratorTag>
void test_find_explicit_sender()
{
    using namespace hpx::execution;

    test_find_explicit_sender(hpx::launch::sync, seq(task), IteratorTag());
    test_find_explicit_sender(hpx::launch::sync, unseq(task), IteratorTag());
    test_find_explicit_sender(hpx::launch::async, par(task), IteratorTag());
    test_find_explicit_sender(
        hpx::launch::async, par_unseq(task), IteratorTag());
}

void find_test_explicit_sender()
{
    test_find_explicit_sender<std::random_access_iterator_tag>();
    test_find_explicit_sender<std::forward_iterator_tag>();
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    find_test_explicit_sender_direct();
    find_test_explicit_sender();

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

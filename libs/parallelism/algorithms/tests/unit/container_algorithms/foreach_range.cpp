//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/init.hpp>
#include <hpx/local/thread.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "foreach_tests.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_for_each()
{
    using namespace hpx::execution;

    test_for_each_seq(IteratorTag());

    test_for_each(seq, IteratorTag());
    test_for_each(par, IteratorTag());
    test_for_each(par_unseq, IteratorTag());

    test_for_each_async(seq(task), IteratorTag());
    test_for_each_async(par(task), IteratorTag());

    test_for_each_sender(seq, IteratorTag());
    test_for_each_sender(par, IteratorTag());
    test_for_each_sender(par_unseq, IteratorTag());
    test_for_each_sender(seq(task), IteratorTag());
    test_for_each_sender(par(task), IteratorTag());
}

void for_each_test()
{
    test_for_each<std::random_access_iterator_tag>();
    test_for_each<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_for_each_exception()
{
    using namespace hpx::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_for_each_exception_seq(IteratorTag());

    test_for_each_exception(seq, IteratorTag());
    test_for_each_exception(par, IteratorTag());

    test_for_each_exception_async(seq(task), IteratorTag());
    test_for_each_exception_async(par(task), IteratorTag());

    test_for_each_exception_sender(seq, IteratorTag());
    test_for_each_exception_sender(par, IteratorTag());
    test_for_each_exception_sender(seq(task), IteratorTag());
    test_for_each_exception_sender(par(task), IteratorTag());
}

void for_each_exception_test()
{
    test_for_each_exception<std::random_access_iterator_tag>();
    test_for_each_exception<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_for_each_bad_alloc()
{
    using namespace hpx::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_for_each_bad_alloc_seq(IteratorTag());

    test_for_each_bad_alloc(seq, IteratorTag());
    test_for_each_bad_alloc(par, IteratorTag());

    test_for_each_bad_alloc_async(seq(task), IteratorTag());
    test_for_each_bad_alloc_async(par(task), IteratorTag());

    test_for_each_bad_alloc_sender(seq, IteratorTag());
    test_for_each_bad_alloc_sender(par, IteratorTag());
    test_for_each_bad_alloc_sender(seq(task), IteratorTag());
    test_for_each_bad_alloc_sender(par(task), IteratorTag());
}

void for_each_bad_alloc_test()
{
    test_for_each_bad_alloc<std::random_access_iterator_tag>();
    test_for_each_bad_alloc<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    for_each_test();
    for_each_exception_test();
    for_each_bad_alloc_test();
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

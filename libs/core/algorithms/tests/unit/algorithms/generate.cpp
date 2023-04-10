//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2020 Hartmut Kaiser
//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "generate_tests.hpp"

// FIXME: Intel 15 currently can not compile this code. This needs to be fixed. See #1408
#if !(defined(HPX_INTEL_VERSION) && HPX_INTEL_VERSION == 1500)
////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_generate()
{
    using namespace hpx::execution;

    test_generate(IteratorTag());

    test_generate(seq, IteratorTag());
    test_generate(par, IteratorTag());
    test_generate(par_unseq, IteratorTag());

    test_generate_async(seq(task), IteratorTag());
    test_generate_async(par(task), IteratorTag());
}

void generate_test()
{
    test_generate<std::random_access_iterator_tag>();
    test_generate<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_generate_exception()
{
    using namespace hpx::execution;

    test_generate_exception(IteratorTag());

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_generate_exception(seq, IteratorTag());
    test_generate_exception(par, IteratorTag());

    test_generate_exception_async(seq(task), IteratorTag());
    test_generate_exception_async(par(task), IteratorTag());
}

void generate_exception_test()
{
    test_generate_exception<std::random_access_iterator_tag>();
    test_generate_exception<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_generate_bad_alloc()
{
    using namespace hpx::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_generate_bad_alloc(seq, IteratorTag());
    test_generate_bad_alloc(par, IteratorTag());

    test_generate_bad_alloc_async(seq(task), IteratorTag());
    test_generate_bad_alloc_async(par(task), IteratorTag());
}

void generate_bad_alloc_test()
{
    test_generate_bad_alloc<std::random_access_iterator_tag>();
    test_generate_bad_alloc<std::forward_iterator_tag>();
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    generate_test();
    generate_exception_test();
    generate_bad_alloc_test();
    return hpx::local::finalize();
}
#else
int hpx_main(hpx::program_options::variables_map& vm)
{
    return hpx::local::finalize();
}
#endif

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

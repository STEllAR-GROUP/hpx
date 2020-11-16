//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/datapar.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "../algorithms/transform_tests.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_transform()
{
    using namespace hpx::execution;

    test_transform(execution::dataseq, IteratorTag());
    test_transform(execution::datapar, IteratorTag());

    test_transform_async(execution::dataseq(task), IteratorTag());
    test_transform_async(execution::datapar(task), IteratorTag());
}

void transform_test()
{
    test_transform<std::random_access_iterator_tag>();
    test_transform<std::forward_iterator_tag>();
}

template <typename IteratorTag>
void test_transform_exception()
{
    using namespace hpx::execution;

    test_transform_exception(execution::dataseq, IteratorTag());
    test_transform_exception(execution::datapar, IteratorTag());

    test_transform_exception_async(execution::dataseq(task), IteratorTag());
    test_transform_exception_async(execution::datapar(task), IteratorTag());
}

void transform_exception_test()
{
    test_transform_exception<std::random_access_iterator_tag>();
    test_transform_exception<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_transform_bad_alloc()
{
    using namespace hpx::execution;

    test_transform_bad_alloc(execution::dataseq, IteratorTag());
    test_transform_bad_alloc(execution::datapar, IteratorTag());

    test_transform_bad_alloc_async(execution::dataseq(task), IteratorTag());
    test_transform_bad_alloc_async(execution::datapar(task), IteratorTag());
}

void transform_bad_alloc_test()
{
    test_transform_bad_alloc<std::random_access_iterator_tag>();
    test_transform_bad_alloc<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    transform_test();
    transform_exception_test();
    transform_bad_alloc_test();
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

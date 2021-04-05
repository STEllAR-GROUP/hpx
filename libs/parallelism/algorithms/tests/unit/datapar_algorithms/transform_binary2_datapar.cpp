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

#include "../algorithms/transform_binary2_tests.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_transform_binary2()
{
    using namespace hpx::execution;

    test_transform_binary2(dataseq, IteratorTag());
    test_transform_binary2(datapar, IteratorTag());

    test_transform_binary2_async(dataseq(task), IteratorTag());
    test_transform_binary2_async(datapar(task), IteratorTag());
}

void transform_binary2_test()
{
    test_transform_binary2<std::random_access_iterator_tag>();
    test_transform_binary2<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_transform_binary2_exception()
{
    using namespace hpx::execution;

    test_transform_binary2_exception(dataseq, IteratorTag());
    test_transform_binary2_exception(datapar, IteratorTag());

    test_transform_binary2_exception_async(dataseq(task), IteratorTag());
    test_transform_binary2_exception_async(datapar(task), IteratorTag());
}

void transform_binary2_exception_test()
{
    test_transform_binary2_exception<std::random_access_iterator_tag>();
    test_transform_binary2_exception<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_transform_binary2_bad_alloc()
{
    using namespace hpx::execution;

    test_transform_binary2_bad_alloc(dataseq, IteratorTag());
    test_transform_binary2_bad_alloc(datapar, IteratorTag());

    test_transform_binary2_bad_alloc_async(dataseq(task), IteratorTag());
    test_transform_binary2_bad_alloc_async(datapar(task), IteratorTag());
}

void transform_binary2_bad_alloc_test()
{
    test_transform_binary2_bad_alloc<std::random_access_iterator_tag>();
    test_transform_binary2_bad_alloc<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    transform_binary2_test();
    transform_binary2_exception_test();
    transform_binary2_bad_alloc_test();
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

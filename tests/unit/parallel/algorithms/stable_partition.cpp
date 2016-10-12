//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "stable_partition_tests.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_stable_partition()
{
    using namespace hpx::parallel;

    test_stable_partition(seq, IteratorTag());
    test_stable_partition(par, IteratorTag());
    test_stable_partition(par_vec, IteratorTag());

    test_stable_partition_async(seq(task), IteratorTag());
    test_stable_partition_async(par(task), IteratorTag());
}

void stable_partition_test()
{
    test_stable_partition<std::random_access_iterator_tag>();
    test_stable_partition<std::bidirectional_iterator_tag>();
}

template <typename IteratorTag>
void test_stable_partition_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_stable_partition_exception(seq, IteratorTag());
    test_stable_partition_exception(par, IteratorTag());

    test_stable_partition_exception_async(seq(task), IteratorTag());
    test_stable_partition_exception_async(par(task), IteratorTag());
}

void stable_partition_exception_test()
{
    test_stable_partition_exception<std::random_access_iterator_tag>();
    test_stable_partition_exception<std::bidirectional_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_stable_partition_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_stable_partition_bad_alloc(seq, IteratorTag());
    test_stable_partition_bad_alloc(par, IteratorTag());

    test_stable_partition_bad_alloc_async(seq(task), IteratorTag());
    test_stable_partition_bad_alloc_async(par(task), IteratorTag());
}

void stable_partition_bad_alloc_test()
{
    test_stable_partition_bad_alloc<std::random_access_iterator_tag>();
    test_stable_partition_bad_alloc<std::bidirectional_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    stable_partition_test();
    stable_partition_exception_test();
    stable_partition_bad_alloc_test();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run")
        ;

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}



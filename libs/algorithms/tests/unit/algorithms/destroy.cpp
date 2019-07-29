//  Copyright (c) 2014-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "destroy_tests.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_destroy()
{
    using namespace hpx::parallel;
    test_destroy(execution::seq, IteratorTag());
    test_destroy(execution::par, IteratorTag());
    test_destroy(execution::par_unseq, IteratorTag());

    test_destroy_async(execution::seq(execution::task),
        IteratorTag());
    test_destroy_async(execution::par(execution::task),
        IteratorTag());
}

void destroy_test()
{
    test_destroy<std::random_access_iterator_tag>();
    test_destroy<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_destroy_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_destroy_exception(execution::seq, IteratorTag());
    test_destroy_exception(execution::par, IteratorTag());

    test_destroy_exception_async(execution::seq(execution::task), IteratorTag());
    test_destroy_exception_async(execution::par(execution::task), IteratorTag());
}

void destroy_exception_test()
{
    test_destroy_exception<std::random_access_iterator_tag>();
    test_destroy_exception<std::forward_iterator_tag>();
}

//////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_destroy_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_destroy_bad_alloc(execution::seq, IteratorTag());
    test_destroy_bad_alloc(execution::par, IteratorTag());

    test_destroy_bad_alloc_async(execution::seq(execution::task), IteratorTag());
    test_destroy_bad_alloc_async(execution::par(execution::task), IteratorTag());
}

void destroy_bad_alloc_test()
{
    test_destroy_bad_alloc<std::random_access_iterator_tag>();
    test_destroy_bad_alloc<std::forward_iterator_tag>();
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    destroy_test();
    destroy_exception_test();
    destroy_bad_alloc_test();
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

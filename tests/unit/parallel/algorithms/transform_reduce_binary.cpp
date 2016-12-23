//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "transform_reduce_binary_tests.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_transform_reduce_binary()
{
    using namespace hpx::parallel;

    test_transform_reduce_binary(execution::seq, IteratorTag());
    test_transform_reduce_binary(execution::par, IteratorTag());
    test_transform_reduce_binary(execution::par_unseq, IteratorTag());

    test_transform_reduce_binary_async(execution::seq(execution::task),
        IteratorTag());
    test_transform_reduce_binary_async(execution::par(execution::task),
        IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_transform_reduce_binary(execution_policy(execution::seq),
        IteratorTag());
    test_transform_reduce_binary(execution_policy(execution::par),
        IteratorTag());
    test_transform_reduce_binary(execution_policy(execution::par_unseq),
        IteratorTag());

    test_transform_reduce_binary(
        execution_policy(execution::seq(execution::task)), IteratorTag());
    test_transform_reduce_binary(
        execution_policy(execution::par(execution::task)), IteratorTag());
#endif
}

void transform_reduce_binary_test()
{
    test_transform_reduce_binary<std::random_access_iterator_tag>();
    test_transform_reduce_binary<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    transform_reduce_binary_test();

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

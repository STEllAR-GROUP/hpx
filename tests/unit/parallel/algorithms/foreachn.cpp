//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "foreach_tests.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_for_each_n()
{
    using namespace hpx::parallel;

    test_for_each_n(execution::seq, IteratorTag());
    test_for_each_n(execution::par, IteratorTag());
    test_for_each_n(execution::par_unseq, IteratorTag());

    test_for_each_n_async(execution::seq(execution::task), IteratorTag());
    test_for_each_n_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_for_each_n(execution_policy(execution::seq), IteratorTag());
    test_for_each_n(execution_policy(execution::par), IteratorTag());
    test_for_each_n(execution_policy(execution::par_unseq), IteratorTag());

    test_for_each_n(execution_policy(execution::seq(execution::task)), IteratorTag());
    test_for_each_n(execution_policy(execution::par(execution::task)), IteratorTag());
#endif
}

void for_each_n_test()
{
    test_for_each_n<std::random_access_iterator_tag>();
    test_for_each_n<std::forward_iterator_tag>();
    test_for_each_n<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    for_each_n_test();
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

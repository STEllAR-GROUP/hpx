//  Copyright (c) 2016 Zahra Khatami
//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "foreach_tests_prefetching.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_prefetching_executors(ExPolicy && policy)
{
    typedef std::random_access_iterator_tag iterator_tag;

    test_for_each_prefetching_exception(policy, iterator_tag());
    test_for_each_prefetching_bad_alloc(policy, iterator_tag());
    test_for_each_prefetching(std::forward<ExPolicy>(policy),
        iterator_tag());
}

template <typename ExPolicy>
void test_prefetching_executors_async(ExPolicy && p)
{
    typedef std::random_access_iterator_tag iterator_tag;

    test_for_each_prefetching_exception_async(p, iterator_tag());
    test_for_each_prefetching_bad_alloc_async(p, iterator_tag());
    test_for_each_prefetching_async(std::forward<ExPolicy>(p),
        iterator_tag());
}

void for_each_prefetching_executors_test()
{
    using namespace hpx::parallel;

    {
        parallel_executor exec;

        test_prefetching_executors(execution::par.on(exec));
        test_prefetching_executors_async(execution::par(execution::task).on(exec));
    }

    {
        sequential_executor exec;
        test_prefetching_executors(execution::par.on(exec));
        test_prefetching_executors_async(execution::par(execution::task).on(exec));
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    for_each_prefetching_executors_test();
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

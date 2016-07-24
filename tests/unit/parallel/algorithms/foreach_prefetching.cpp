//  Copyright (c) 2016 Zahra Khatami
//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <string>
#include <vector>

#include "foreach_tests_prefetching.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_for_each_prefetching()
{
    using namespace hpx::parallel;

    test_for_each_prefetching(par, IteratorTag());
    test_for_each_prefetching(par_vec, IteratorTag());
    test_for_each_prefetching_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_for_each_prefetching(execution_policy(par),
        IteratorTag());
    test_for_each_prefetching(execution_policy(par_vec),
        IteratorTag());
    test_for_each_prefetching(execution_policy(par(task)),
        IteratorTag());
#endif
}

void for_each_prefetching_test()
{
    test_for_each_prefetching<std::random_access_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_for_each_prefetching_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_for_each_prefetching_exception(par, IteratorTag());
    test_for_each_prefetching_exception_async(par(task),
        IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_for_each_prefetching_exception(execution_policy(par),
        IteratorTag());
    test_for_each_prefetching_exception(execution_policy(par(task)),
        IteratorTag());
#endif
}

void for_each_prefetching_exception_test()
{
    test_for_each_prefetching_exception<std::random_access_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_for_each_prefetching_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_for_each_prefetching_bad_alloc(par, IteratorTag());
    test_for_each_prefetching_bad_alloc_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_for_each_prefetching_bad_alloc(execution_policy(par),
        IteratorTag());
    test_for_each_prefetching_bad_alloc(execution_policy(par(task)),
        IteratorTag());
#endif
}

void for_each_prefetching_bad_alloc_test()
{
    test_for_each_prefetching_bad_alloc<std::random_access_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    for_each_prefetching_test();
    for_each_prefetching_exception_test();
    for_each_prefetching_bad_alloc_test();
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
        "the random number generator seed to use for this run");

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

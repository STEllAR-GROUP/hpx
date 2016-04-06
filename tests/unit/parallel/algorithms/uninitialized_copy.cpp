//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include "uninitialized_copy_tests.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_uninitialized_copy()
{
    using namespace hpx::parallel;
    test_uninitialized_copy(seq, IteratorTag());
    test_uninitialized_copy(par, IteratorTag());
    test_uninitialized_copy(par_vec, IteratorTag());

    test_uninitialized_copy_async(seq(task), IteratorTag());
    test_uninitialized_copy_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_uninitialized_copy(execution_policy(seq), IteratorTag());
    test_uninitialized_copy(execution_policy(par), IteratorTag());
    test_uninitialized_copy(execution_policy(par_vec), IteratorTag());

    test_uninitialized_copy(execution_policy(seq(task)), IteratorTag());
    test_uninitialized_copy(execution_policy(par(task)), IteratorTag());
#endif
}

void uninitialized_copy_test()
{
    test_uninitialized_copy<std::random_access_iterator_tag>();
    test_uninitialized_copy<std::forward_iterator_tag>();
    test_uninitialized_copy<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_uninitialized_copy_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_uninitialized_copy_exception(seq, IteratorTag());
    test_uninitialized_copy_exception(par, IteratorTag());

    test_uninitialized_copy_exception_async(seq(task), IteratorTag());
    test_uninitialized_copy_exception_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_uninitialized_copy_exception(execution_policy(seq), IteratorTag());
    test_uninitialized_copy_exception(execution_policy(par), IteratorTag());

    test_uninitialized_copy_exception(execution_policy(seq(task)), IteratorTag());
    test_uninitialized_copy_exception(execution_policy(par(task)), IteratorTag());
#endif
}

void uninitialized_copy_exception_test()
{
    test_uninitialized_copy_exception<std::random_access_iterator_tag>();
    test_uninitialized_copy_exception<std::forward_iterator_tag>();
    test_uninitialized_copy_exception<std::input_iterator_tag>();
}

//////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_uninitialized_copy_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_uninitialized_copy_bad_alloc(seq, IteratorTag());
    test_uninitialized_copy_bad_alloc(par, IteratorTag());

    test_uninitialized_copy_bad_alloc_async(seq(task), IteratorTag());
    test_uninitialized_copy_bad_alloc_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_uninitialized_copy_bad_alloc(execution_policy(seq), IteratorTag());
    test_uninitialized_copy_bad_alloc(execution_policy(par), IteratorTag());

    test_uninitialized_copy_bad_alloc(execution_policy(seq(task)), IteratorTag());
    test_uninitialized_copy_bad_alloc(execution_policy(par(task)), IteratorTag());
#endif
}

void uninitialized_copy_bad_alloc_test()
{
    test_uninitialized_copy_bad_alloc<std::random_access_iterator_tag>();
    test_uninitialized_copy_bad_alloc<std::forward_iterator_tag>();
    test_uninitialized_copy_bad_alloc<std::input_iterator_tag>();
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    uninitialized_copy_test();
    uninitialized_copy_exception_test();
    uninitialized_copy_bad_alloc_test();
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
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        std::to_string(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

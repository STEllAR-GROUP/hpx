//  Copyright (c) 2014-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "uninitialized_move_tests.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_uninitialized_move()
{
    using namespace hpx::parallel;
    test_uninitialized_move(execution::seq, IteratorTag());
    test_uninitialized_move(execution::par, IteratorTag());
    test_uninitialized_move(execution::par_unseq, IteratorTag());

    test_uninitialized_move_async(execution::seq(execution::task),
        IteratorTag());
    test_uninitialized_move_async(execution::par(execution::task),
        IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_uninitialized_move(execution_policy(execution::seq), IteratorTag());
    test_uninitialized_move(execution_policy(execution::par), IteratorTag());
    test_uninitialized_move(execution_policy(execution::par_unseq),
        IteratorTag());

    test_uninitialized_move(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_uninitialized_move(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void uninitialized_move_test()
{
    test_uninitialized_move<std::random_access_iterator_tag>();
    test_uninitialized_move<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_uninitialized_move<std::input_iterator_tag>();
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_uninitialized_move_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_uninitialized_move_exception(execution::seq, IteratorTag());
    test_uninitialized_move_exception(execution::par, IteratorTag());

    test_uninitialized_move_exception_async(execution::seq(execution::task),
        IteratorTag());
    test_uninitialized_move_exception_async(execution::par(execution::task),
        IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_uninitialized_move_exception(execution_policy(execution::seq),
        IteratorTag());
    test_uninitialized_move_exception(execution_policy(execution::par),
        IteratorTag());

    test_uninitialized_move_exception(
        execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_uninitialized_move_exception(
        execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void uninitialized_move_exception_test()
{
    test_uninitialized_move_exception<std::random_access_iterator_tag>();
    test_uninitialized_move_exception<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_uninitialized_move_exception<std::input_iterator_tag>();
#endif
}

//////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_uninitialized_move_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_uninitialized_move_bad_alloc(execution::seq, IteratorTag());
    test_uninitialized_move_bad_alloc(execution::par, IteratorTag());

    test_uninitialized_move_bad_alloc_async(execution::seq(execution::task),
        IteratorTag());
    test_uninitialized_move_bad_alloc_async(execution::par(execution::task),
        IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_uninitialized_move_bad_alloc(execution_policy(execution::seq),
        IteratorTag());
    test_uninitialized_move_bad_alloc(execution_policy(execution::par),
        IteratorTag());

    test_uninitialized_move_bad_alloc(
        execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_uninitialized_move_bad_alloc(
        execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void uninitialized_move_bad_alloc_test()
{
    test_uninitialized_move_bad_alloc<std::random_access_iterator_tag>();
    test_uninitialized_move_bad_alloc<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_uninitialized_move_bad_alloc<std::input_iterator_tag>();
#endif
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    uninitialized_move_test();
    uninitialized_move_exception_test();
    uninitialized_move_bad_alloc_test();
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

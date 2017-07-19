//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "inclusive_scan_tests.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_inclusive_scan1()
{
    using namespace hpx::parallel;

    test_inclusive_scan1(execution::seq, IteratorTag());
    test_inclusive_scan1(execution::par, IteratorTag());
    test_inclusive_scan1(execution::par_unseq, IteratorTag());

    test_inclusive_scan1_async(execution::seq(execution::task), IteratorTag());
    test_inclusive_scan1_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_inclusive_scan1(execution_policy(execution::seq), IteratorTag());
    test_inclusive_scan1(execution_policy(execution::par), IteratorTag());
    test_inclusive_scan1(execution_policy(execution::par_unseq), IteratorTag());

    test_inclusive_scan1(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_inclusive_scan1(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void inclusive_scan_test1()
{
    test_inclusive_scan1<std::random_access_iterator_tag>();
    test_inclusive_scan1<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_inclusive_scan1<std::input_iterator_tag>();
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_inclusive_scan2()
{
    using namespace hpx::parallel;

    test_inclusive_scan2(execution::seq, IteratorTag());
    test_inclusive_scan2(execution::par, IteratorTag());
    test_inclusive_scan2(execution::par_unseq, IteratorTag());

    test_inclusive_scan2_async(execution::seq(execution::task), IteratorTag());
    test_inclusive_scan2_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_inclusive_scan2(execution_policy(execution::seq), IteratorTag());
    test_inclusive_scan2(execution_policy(execution::par), IteratorTag());
    test_inclusive_scan2(execution_policy(execution::par_unseq), IteratorTag());

    test_inclusive_scan2(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_inclusive_scan2(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void inclusive_scan_test2()
{
    test_inclusive_scan2<std::random_access_iterator_tag>();
    test_inclusive_scan2<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_inclusive_scan2<std::input_iterator_tag>();
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_inclusive_scan3()
{
    using namespace hpx::parallel;

    test_inclusive_scan3(execution::seq, IteratorTag());
    test_inclusive_scan3(execution::par, IteratorTag());
    test_inclusive_scan3(execution::par_unseq, IteratorTag());

    test_inclusive_scan3_async(execution::seq(execution::task), IteratorTag());
    test_inclusive_scan3_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_inclusive_scan3(execution_policy(execution::seq), IteratorTag());
    test_inclusive_scan3(execution_policy(execution::par), IteratorTag());
    test_inclusive_scan3(execution_policy(execution::par_unseq), IteratorTag());

    test_inclusive_scan3(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_inclusive_scan3(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void inclusive_scan_test3()
{
    test_inclusive_scan3<std::random_access_iterator_tag>();
    test_inclusive_scan3<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_inclusive_scan3<std::input_iterator_tag>();
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_inclusive_scan_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_inclusive_scan_exception(execution::seq, IteratorTag());
    test_inclusive_scan_exception(execution::par, IteratorTag());

    test_inclusive_scan_exception_async(execution::seq(execution::task),
        IteratorTag());
    test_inclusive_scan_exception_async(execution::par(execution::task),
        IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_inclusive_scan_exception(execution_policy(execution::seq), IteratorTag());
    test_inclusive_scan_exception(execution_policy(execution::par), IteratorTag());

    test_inclusive_scan_exception(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_inclusive_scan_exception(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void inclusive_scan_exception_test()
{
    test_inclusive_scan_exception<std::random_access_iterator_tag>();
    test_inclusive_scan_exception<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_inclusive_scan_exception<std::input_iterator_tag>();
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_inclusive_scan_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_inclusive_scan_bad_alloc(execution::seq, IteratorTag());
    test_inclusive_scan_bad_alloc(execution::par, IteratorTag());

    test_inclusive_scan_bad_alloc_async(execution::seq(execution::task),
        IteratorTag());
    test_inclusive_scan_bad_alloc_async(execution::par(execution::task),
        IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_inclusive_scan_bad_alloc(execution_policy(execution::seq), IteratorTag());
    test_inclusive_scan_bad_alloc(execution_policy(execution::par), IteratorTag());

    test_inclusive_scan_bad_alloc(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_inclusive_scan_bad_alloc(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void inclusive_scan_bad_alloc_test()
{
    test_inclusive_scan_bad_alloc<std::random_access_iterator_tag>();
    test_inclusive_scan_bad_alloc<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_inclusive_scan_bad_alloc<std::input_iterator_tag>();
#endif
}

////////////////////////////////////////////////////////////////////////////////
void inclusive_scan_validate()
{
    std::vector<int> a, b;
    // test scan algorithms using separate array for output
    //  std::cout << " Validating dual arrays " <<std::endl;
    test_inclusive_scan_validate(hpx::parallel::execution::seq, a, b);
    test_inclusive_scan_validate(hpx::parallel::execution::par, a, b);
    // test scan algorithms using same array for input and output
    //  std::cout << " Validating in_place arrays " <<std::endl;
    test_inclusive_scan_validate(hpx::parallel::execution::seq, a, a);
    test_inclusive_scan_validate(hpx::parallel::execution::par, a, a);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    // if benchmark is requested we run it even in debug mode
    if (vm.count("benchmark")) {
        inclusive_scan_benchmark();
    }
    else {
        inclusive_scan_test1();
        inclusive_scan_test2();
        inclusive_scan_test3();

        inclusive_scan_exception_test();
        inclusive_scan_bad_alloc_test();

        inclusive_scan_validate();
#ifndef HPX_DEBUG
        inclusive_scan_benchmark();
#endif
    }

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
        ("benchmark", "run a timing benchmark only");

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

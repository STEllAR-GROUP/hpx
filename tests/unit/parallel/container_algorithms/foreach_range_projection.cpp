//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/threads.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "foreach_tests_projection.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag, typename Proj>
void test_for_each()
{
    using namespace hpx::parallel;

    test_for_each(execution::seq, IteratorTag(), Proj());
    test_for_each(execution::par, IteratorTag(), Proj());
    test_for_each(execution::par_unseq, IteratorTag(), Proj());

    test_for_each_async(execution::seq(execution::task), IteratorTag(), Proj());
    test_for_each_async(execution::par(execution::task), IteratorTag(), Proj());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_for_each(execution_policy(execution::seq), IteratorTag(), Proj());
    test_for_each(execution_policy(execution::par), IteratorTag(), Proj());
    test_for_each(execution_policy(execution::par_unseq), IteratorTag(), Proj());

    test_for_each(execution_policy(execution::seq(execution::task)), IteratorTag(), Proj());
    test_for_each(execution_policy(execution::par(execution::task)), IteratorTag(), Proj());
#endif
}

template <typename Proj>
void for_each_test()
{
    test_for_each<std::random_access_iterator_tag, Proj>();
    test_for_each<std::forward_iterator_tag, Proj>();
    test_for_each<std::input_iterator_tag, Proj>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag, typename Proj>
void test_for_each_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_for_each_exception(execution::seq, IteratorTag(), Proj());
    test_for_each_exception(execution::par, IteratorTag(), Proj());

    test_for_each_exception_async(execution::seq(execution::task), IteratorTag(), Proj());
    test_for_each_exception_async(execution::par(execution::task), IteratorTag(), Proj());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_for_each_exception(execution_policy(execution::seq),
        IteratorTag(), Proj());
    test_for_each_exception(execution_policy(execution::par),
        IteratorTag(), Proj());
    test_for_each_exception(execution_policy(execution::seq(execution::task)),
        IteratorTag(), Proj());
    test_for_each_exception(execution_policy(execution::par(execution::task)),
        IteratorTag(), Proj());
#endif
}

template <typename Proj>
void for_each_exception_test()
{
    test_for_each_exception<std::random_access_iterator_tag, Proj>();
    test_for_each_exception<std::forward_iterator_tag, Proj>();
    test_for_each_exception<std::input_iterator_tag, Proj>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag, typename Proj>
void test_for_each_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_for_each_bad_alloc(execution::seq, IteratorTag(), Proj());
    test_for_each_bad_alloc(execution::par, IteratorTag(), Proj());

    test_for_each_bad_alloc_async(execution::seq(execution::task), IteratorTag(), Proj());
    test_for_each_bad_alloc_async(execution::par(execution::task), IteratorTag(), Proj());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_for_each_bad_alloc(execution_policy(execution::seq),
        IteratorTag(), Proj());
    test_for_each_bad_alloc(execution_policy(execution::par),
        IteratorTag(), Proj());
    test_for_each_bad_alloc(execution_policy(execution::seq(execution::task)),
        IteratorTag(), Proj());
    test_for_each_bad_alloc(execution_policy(execution::par(execution::task)),
        IteratorTag(), Proj());
#endif
}

template <typename Proj>
void for_each_bad_alloc_test()
{
    test_for_each_bad_alloc<std::random_access_iterator_tag, Proj>();
    test_for_each_bad_alloc<std::forward_iterator_tag, Proj>();
    test_for_each_bad_alloc<std::input_iterator_tag, Proj>();
}

///////////////////////////////////////////////////////////////////////////////
struct projection_square
{
    template <typename T>
    T operator()(T const& val) const
    {
        return val * val;
    }
};

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    for_each_test<hpx::parallel::util::projection_identity>();
    for_each_exception_test<hpx::parallel::util::projection_identity>();
    for_each_bad_alloc_test<hpx::parallel::util::projection_identity>();

    for_each_test<projection_square>();
    for_each_exception_test<projection_square>();
    for_each_bad_alloc_test<projection_square>();

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

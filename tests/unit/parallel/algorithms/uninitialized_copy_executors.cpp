
//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include "uninitialized_copy_tests.hpp"

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_executors(ExPolicy && policy)
{
    typedef std::random_access_iterator_tag iterator_tag;

    test_uninitialized_copy(policy, iterator_tag());
    test_uninitialized_copy_exception(policy, iterator_tag());
    test_uninitialized_copy_bad_alloc(policy, iterator_tag());
    test_uninitialized_copy(
        std::forward<ExPolicy>(policy), iterator_tag());
}

template <typename ExPolicy>
void test_executors_async(ExPolicy && p)
{
    typedef std::random_access_iterator_tag iterator_tag;

    test_uninitialized_copy_async(p, iterator_tag());
    test_uninitialized_copy_exception_async(p, iterator_tag());
    test_uninitialized_copy_bad_alloc_async(p, iterator_tag());
    test_uninitialized_copy_async(
        std::forward<ExPolicy>(p), iterator_tag());
}

void uninitialized_copy_executors_test()
{
    using namespace hpx::parallel;

    {
        parallel_executor exec;

        test_executors(par.on(exec));
        test_executors_async(par(task).on(exec));
    }

    {
        sequential_executor exec;

        test_executors(seq.on(exec));
        test_executors_async(seq(task).on(exec));

        test_executors(par.on(exec));
        test_executors_async(par(task).on(exec));
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

    uninitialized_copy_executors_test();
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
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

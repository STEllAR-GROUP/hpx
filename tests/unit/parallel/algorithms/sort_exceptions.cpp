//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

// use smaller array sizes for debug tests
#if defined(HPX_DEBUG)
#define HPX_SORT_TEST_SIZE          50000
#define HPX_SORT_TEST_SIZE_STRINGS  10000
#endif

#include "sort_tests.hpp"

///////////////////////////////////////////////////////////////////////////////
void test_exceptions()
{
    using namespace hpx::parallel;

    // default comparison operator (std::less)
    test_sort_exception(seq,     int());
    test_sort_exception(par,     int());

    // user supplied comparison operator (std::less)
    test_sort_exception(seq,     int(), std::less<int>());
    test_sort_exception(par,     int(), std::less<int>());

    // Async execution, default comparison operator
    test_sort_exception_async(seq(task), int());
    test_sort_exception_async(par(task), int());

    // Async execution, user comparison operator
    test_sort_exception_async(seq(task), int(),  std::less<int>());
    test_sort_exception_async(par(task), int(), std::less<int>());
}

////////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test_exceptions();
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

    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

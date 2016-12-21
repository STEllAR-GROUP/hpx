//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

// use smaller array sizes for debug tests
#if defined(HPX_DEBUG)
#define HPX_SORT_TEST_SIZE          50000L
#define HPX_SORT_TEST_SIZE_STRINGS  10000L
#endif

#include "sort_tests.hpp"

////////////////////////////////////////////////////////////////////////////////
// this function times a sort and outputs the time for cDash to plot it
void sort_benchmark()
{
    try {
        using namespace hpx::parallel;
        // Fill vector with random values
        std::vector<double> c(HPX_SORT_TEST_SIZE << 4);
        rnd_fill<double>(c, (std::numeric_limits<double>::min)(),
            (std::numeric_limits<double>::max)(), double(std::rand()));

        hpx::util::high_resolution_timer t;
        // sort, blocking when seq, par, par_vec
        hpx::parallel::sort(execution::par, c.begin(), c.end());
        double elapsed = t.elapsed();

        bool is_sorted = (verify(c, std::less<double>(), elapsed, true)!=0);
        HPX_TEST(is_sorted);
        if (is_sorted) {
            std::cout << "<DartMeasurement name=\"SortDoublesTime\" \n"
                << "type=\"numeric/double\">" << elapsed << "</DartMeasurement> \n";
        }
    }
    catch (...) {
        HPX_TEST(false);
    }
}

////////////////////////////////////////////////////////////////////////////////
void test_sort1()
{
    using namespace hpx::parallel;

    // default comparison operator (std::less)
    test_sort1(execution::seq,     int());
    test_sort1(execution::par,     int());
    test_sort1(execution::par_unseq, int());

    // default comparison operator (std::less)
    test_sort1(execution::seq,     double());
    test_sort1(execution::par,     double());
    test_sort1(execution::par_unseq, double());

    // default comparison operator (std::less)
    test_sort1(execution::seq,     std::string());
    test_sort1(execution::par,     std::string());
    test_sort1(execution::par_unseq, std::string());

    // user supplied comparison operator (std::less)
    test_sort1_comp(execution::seq,     int(), std::less<std::size_t>());
    test_sort1_comp(execution::par,     int(), std::less<std::size_t>());
    test_sort1_comp(execution::par_unseq, int(), std::less<std::size_t>());

    // user supplied comparison operator (std::greater)
    test_sort1_comp(execution::seq,     double(), std::greater<double>());
    test_sort1_comp(execution::par,     double(), std::greater<double>());
    test_sort1_comp(execution::par_unseq, double(), std::greater<double>());

    // default comparison operator (std::less)
    test_sort1_comp(execution::seq,     std::string(), std::greater<std::string>());
    test_sort1_comp(execution::par,     std::string(), std::greater<std::string>());
    test_sort1_comp(execution::par_unseq, std::string(), std::greater<std::string>());

    // Async execution, default comparison operator
    test_sort1_async(execution::seq(execution::task), int());
    test_sort1_async(execution::par(execution::task), char());
    test_sort1_async(execution::seq(execution::task), double());
    test_sort1_async(execution::par(execution::task), float());
    test_sort1_async_str(execution::seq(execution::task));
    test_sort1_async_str(execution::par(execution::task));

    // Async execution, user comparison operator
    test_sort1_async(execution::seq(execution::task), int(),    std::less<unsigned int>());
    test_sort1_async(execution::par(execution::task), char(),   std::less<char>());
    //
    test_sort1_async(execution::seq(execution::task), double(), std::greater<double>());
    test_sort1_async(execution::par(execution::task), float(),  std::greater<float>());
    //
    test_sort1_async_str(execution::seq(execution::task), std::greater<std::string>());
    test_sort1_async_str(execution::par(execution::task), std::greater<std::string>());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_sort1(execution_policy(execution::seq),       int());
    test_sort1(execution_policy(execution::par),       int());
    test_sort1(execution_policy(execution::par_unseq),   int());
    test_sort1(execution_policy(execution::seq(execution::task)), int());
    test_sort1(execution_policy(execution::par(execution::task)), int());
    test_sort1(execution_policy(execution::seq(execution::task)), std::string());
    test_sort1(execution_policy(execution::par(execution::task)), std::string());
#endif
}

void test_sort2()
{
    using namespace hpx::parallel;
    // default comparison operator (std::less)
    test_sort2(execution::seq,     int());
    test_sort2(execution::par,     int());
    test_sort2(execution::par_unseq, int());

    // default comparison operator (std::less)
    test_sort2(execution::seq,     double());
    test_sort2(execution::par,     double());
    test_sort2(execution::par_unseq, double());

    // user supplied comparison operator (std::less)
    test_sort2_comp(execution::seq,     int(), std::less<std::size_t>());
    test_sort2_comp(execution::par,     int(), std::less<std::size_t>());
    test_sort2_comp(execution::par_unseq, int(), std::less<std::size_t>());

    // user supplied comparison operator (std::greater)
    test_sort2_comp(execution::seq,     double(), std::greater<double>());
    test_sort2_comp(execution::par,     double(), std::greater<double>());
    test_sort2_comp(execution::par_unseq, double(), std::greater<double>());

    // Async execution, default comparison operator
    test_sort2_async(execution::seq(execution::task), int());
    test_sort2_async(execution::par(execution::task), char());
    test_sort2_async(execution::seq(execution::task), double());
    test_sort2_async(execution::par(execution::task), float());

    // Async execution, user comparison operator
    test_sort2_async(execution::seq(execution::task), int(),    std::less<unsigned int>());
    test_sort2_async(execution::par(execution::task), char(),   std::less<char>());
    //
    test_sort2_async(execution::seq(execution::task), double(), std::greater<double>());
    test_sort2_async(execution::par(execution::task), float(),  std::greater<float>());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_sort2(execution_policy(execution::seq),       int());
    test_sort2(execution_policy(execution::par),       int());
    test_sort2(execution_policy(execution::par_unseq),   int());
    test_sort2(execution_policy(execution::seq(execution::task)), int());
    test_sort2(execution_policy(execution::par(execution::task)), int());
#endif
}

////////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    // if benchmark is requested we run it even in debug mode
    if (vm.count("benchmark")) {
        sort_benchmark();
    }
    else {
        test_sort1();
        test_sort2();
#ifndef HPX_DEBUG
        sort_benchmark();
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

    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

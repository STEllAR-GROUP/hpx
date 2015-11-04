//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include "sort_tests.hpp"

////////////////////////////////////////////////////////////////////////////////
void test_sort1()
{
    using namespace hpx::parallel;

    // default comparison operator (std::less)
    test_sort1(seq,     int());
    test_sort1(par,     int());
    test_sort1(par_vec, int());

    // default comparison operator (std::less)
    test_sort1(seq,     double());
    test_sort1(par,     double());
    test_sort1(par_vec, double());

    // user supplied comparison operator (std::less)
    test_sort1_comp(seq,     int(), std::less<std::size_t>());
    test_sort1_comp(par,     int(), std::less<std::size_t>());
    test_sort1_comp(par_vec, int(), std::less<std::size_t>());

    // user supplied comparison operator (std::greater)
    test_sort1_comp(seq,     double(), std::greater<std::size_t>());
    test_sort1_comp(par,     double(), std::greater<std::size_t>());
    test_sort1_comp(par_vec, double(), std::greater<std::size_t>());

    // Async execution, default comparison operator
    test_sort1_async(seq(task), int());
    test_sort1_async(par(task), char());
    test_sort1_async(seq(task), double());
    test_sort1_async(par(task), float());

    // Async execution, user comparison operator
    test_sort1_async(seq(task), int(),    std::less<unsigned int>());
    test_sort1_async(par(task), char(),   std::less<char>());
    //
    test_sort1_async(seq(task), double(), std::greater<double>());
    test_sort1_async(par(task), float(),  std::greater<float>());

    test_sort1(execution_policy(seq),       int());
    test_sort1(execution_policy(par),       int());
    test_sort1(execution_policy(par_vec),   int());
    test_sort1(execution_policy(seq(task)), int());
    test_sort1(execution_policy(par(task)), int());
}

void test_sort2()
{
    using namespace hpx::parallel;
    // default comparison operator (std::less)
    test_sort2(seq,     int());
    test_sort2(par,     int());
    test_sort2(par_vec, int());

    // default comparison operator (std::less)
    test_sort2(seq,     double());
    test_sort2(par,     double());
    test_sort2(par_vec, double());

    // user supplied comparison operator (std::less)
    test_sort2_comp(seq,     int(), std::less<std::size_t>());
    test_sort2_comp(par,     int(), std::less<std::size_t>());
    test_sort2_comp(par_vec, int(), std::less<std::size_t>());

    // user supplied comparison operator (std::greater)
    test_sort2_comp(seq,     double(), std::greater<std::size_t>());
    test_sort2_comp(par,     double(), std::greater<std::size_t>());
    test_sort2_comp(par_vec, double(), std::greater<std::size_t>());

    // Async execution, default comparison operator
    test_sort2_async(seq(task), int());
    test_sort2_async(par(task), char());
    test_sort2_async(seq(task), double());
    test_sort2_async(par(task), float());

    // Async execution, user comparison operator
    test_sort2_async(seq(task), int(),    std::less<unsigned int>());
    test_sort2_async(par(task), char(),   std::less<char>());
    //
    test_sort2_async(seq(task), double(), std::greater<double>());
    test_sort2_async(par(task), float(),  std::greater<float>());

    test_sort2(execution_policy(seq),       int());
    test_sort2(execution_policy(par),       int());
    test_sort2(execution_policy(par_vec),   int());
    test_sort2(execution_policy(seq(task)), int());
    test_sort2(execution_policy(par(task)), int());
}

////////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    test_sort1();
    test_sort2();
//    sorted_exception_test();
//    sorted_bad_alloc_test();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

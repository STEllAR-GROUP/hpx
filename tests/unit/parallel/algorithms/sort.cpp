//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include "sort_tests.hpp"

#define msg(a,b,c) \
  std::cout << std::setw(12) << #a << std::setw(4) << #b << std::setw(6) << #c << "\t";

////////////////////////////////////////////////////////////////////////////////
void test_sort1()
{
    using namespace hpx::parallel;

    // default comparison operator (std::less)
    msg(seq,      ., sync); test_sort1(seq);
    msg(par,      ., sync); test_sort1(par);
    msg(par_vec,  ., sync); test_sort1(par_vec);

    // user supplied comparison operator (std::less)
    msg(seq,     <, sync); test_sort1_comp(seq, std::less<std::size_t>());
    msg(par,     <, sync); test_sort1_comp(par, std::less<std::size_t>());
    msg(par_vec, <, sync); test_sort1_comp(par_vec, std::less<std::size_t>());

    // user supplied comparison operator (std::greater)
    msg(seq,     >, sync); test_sort1_comp(seq, std::greater<std::size_t>());
    msg(par,     >, sync); test_sort1_comp(par, std::greater<std::size_t>());
    msg(par_vec, >, sync); test_sort1_comp(par_vec, std::greater<std::size_t>());

    // Async execution, default comparison operator
    msg(seq_task, <, async); test_sort1_async(seq(task), std::less<std::size_t>());
    // can't use async task execution yet because we need to keep internal sort
    // structure allocated inside dispatch alive until the function completes
    msg(par_task, <, async); test_sort1_async(par(task), std::less<std::size_t>());


    msg(exe_seq,      ., sync); test_sort1(execution_policy(seq));
    msg(exe_par,      ., sync); test_sort1(execution_policy(par));
    msg(exe_par_vec,  ., sync); test_sort1(execution_policy(par_vec));
    msg(exe_seq_task, ., async); test_sort1(execution_policy(seq(task)));
    msg(exe_par_task, ., async); test_sort1(execution_policy(par(task)));

}

void sort_test1()
{
    test_sort1();
}

////////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    sort_test1();
 //   sorted_test2();
 //   sorted_test3();
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

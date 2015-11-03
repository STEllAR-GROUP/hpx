//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include "sort_tests.hpp"


////////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_sort1()
{
    using namespace hpx::parallel;
    test_sort1(seq, IteratorTag());
    test_sort1(par, IteratorTag());
    test_sort1(par_vec, IteratorTag());

    test_sort1_async(seq(task), IteratorTag());
    test_sort1_async(par(task), IteratorTag());


    test_sort1(execution_policy(seq), IteratorTag());
    test_sort1(execution_policy(par), IteratorTag());
    test_sort1(execution_policy(par_vec), IteratorTag());
    test_sort1(execution_policy(seq(task)), IteratorTag());
    test_sort1(execution_policy(par(task)), IteratorTag());
}

void sort_test1()
{
    test_sort1<std::random_access_iterator_tag>();
    test_sort1<std::forward_iterator_tag>();
}
/*
////////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_sorted2()
{
    using namespace hpx::parallel;
    test_sorted2(seq, IteratorTag());
    test_sorted2(par, IteratorTag());
    test_sorted2(par_vec, IteratorTag());

    test_sorted2_async(seq(task), IteratorTag());
    test_sorted2_async(par(task), IteratorTag());


    test_sorted2(execution_policy(seq), IteratorTag());
    test_sorted2(execution_policy(par), IteratorTag());
    test_sorted2(execution_policy(par_vec), IteratorTag());
    test_sorted2(execution_policy(seq(task)), IteratorTag());
    test_sorted2(execution_policy(par(task)), IteratorTag());
}

void sorted_test2()
{
    test_sorted2<std::random_access_iterator_tag>();
    test_sorted2<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_sorted3()
{
    using namespace hpx::parallel;
    test_sorted3(seq, IteratorTag());
    test_sorted3(par, IteratorTag());
    test_sorted3(par_vec, IteratorTag());

    test_sorted3_async(seq(task), IteratorTag());
    test_sorted3_async(par(task), IteratorTag());


    test_sorted3(execution_policy(seq), IteratorTag());
    test_sorted3(execution_policy(par), IteratorTag());
    test_sorted3(execution_policy(par_vec), IteratorTag());
    test_sorted3(execution_policy(seq(task)), IteratorTag());
    test_sorted3(execution_policy(par(task)), IteratorTag());
}

void sorted_test3()
{
    test_sorted3<std::random_access_iterator_tag>();
    test_sorted3<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_sorted_exception()
{
    using namespace hpx::parallel;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. Therefore we do not test exceptions
    //  with a vector execution policy
    test_sorted_exception(seq, IteratorTag());
    test_sorted_exception(par, IteratorTag());

    test_sorted_exception_async(seq(task), IteratorTag());
    test_sorted_exception_async(par(task), IteratorTag());

    test_sorted_exception(execution_policy(par), IteratorTag());
    test_sorted_exception(execution_policy(seq(task)), IteratorTag());
    test_sorted_exception(execution_policy(par(task)), IteratorTag());
}
void sorted_exception_test()
{
    test_sorted_exception<std::random_access_iterator_tag>();
    test_sorted_exception<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_sorted_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_sorted_bad_alloc(par, IteratorTag());
    test_sorted_bad_alloc(seq, IteratorTag());

    test_sorted_bad_alloc_async(seq(task), IteratorTag());
    test_sorted_bad_alloc_async(par(task), IteratorTag());

    test_sorted_bad_alloc(execution_policy(par), IteratorTag());
    test_sorted_bad_alloc(execution_policy(seq), IteratorTag());
    test_sorted_bad_alloc(execution_policy(seq(task)), IteratorTag());
    test_sorted_bad_alloc(execution_policy(par(task)), IteratorTag());
}

void sorted_bad_alloc_test()
{
    test_sorted_bad_alloc<std::random_access_iterator_tag>();
    test_sorted_bad_alloc<std::forward_iterator_tag>();
}
*/
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

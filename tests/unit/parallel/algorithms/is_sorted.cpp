//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "is_sorted_tests.hpp"

////////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_sorted1()
{
    using namespace hpx::parallel;
    test_sorted1(execution::seq, IteratorTag());
    test_sorted1(execution::par, IteratorTag());
    test_sorted1(execution::par_unseq, IteratorTag());

    test_sorted1_async(execution::seq(execution::task), IteratorTag());
    test_sorted1_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_sorted1(execution_policy(execution::seq), IteratorTag());
    test_sorted1(execution_policy(execution::par), IteratorTag());
    test_sorted1(execution_policy(execution::par_unseq), IteratorTag());
    test_sorted1(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_sorted1(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void sorted_test1()
{
    test_sorted1<std::random_access_iterator_tag>();
    test_sorted1<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_sorted2()
{
    using namespace hpx::parallel;
    test_sorted2(execution::seq, IteratorTag());
    test_sorted2(execution::par, IteratorTag());
    test_sorted2(execution::par_unseq, IteratorTag());

    test_sorted2_async(execution::seq(execution::task), IteratorTag());
    test_sorted2_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_sorted2(execution_policy(execution::seq), IteratorTag());
    test_sorted2(execution_policy(execution::par), IteratorTag());
    test_sorted2(execution_policy(execution::par_unseq), IteratorTag());
    test_sorted2(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_sorted2(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
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
    test_sorted3(execution::seq, IteratorTag());
    test_sorted3(execution::par, IteratorTag());
    test_sorted3(execution::par_unseq, IteratorTag());

    test_sorted3_async(execution::seq(execution::task), IteratorTag());
    test_sorted3_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_sorted3(execution_policy(execution::seq), IteratorTag());
    test_sorted3(execution_policy(execution::par), IteratorTag());
    test_sorted3(execution_policy(execution::par_unseq), IteratorTag());
    test_sorted3(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_sorted3(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
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
    test_sorted_exception(execution::seq, IteratorTag());
    test_sorted_exception(execution::par, IteratorTag());

    test_sorted_exception_async(execution::seq(execution::task), IteratorTag());
    test_sorted_exception_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_sorted_exception(execution_policy(execution::par), IteratorTag());
    test_sorted_exception(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_sorted_exception(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
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
    test_sorted_bad_alloc(execution::par, IteratorTag());
    test_sorted_bad_alloc(execution::seq, IteratorTag());

    test_sorted_bad_alloc_async(execution::seq(execution::task), IteratorTag());
    test_sorted_bad_alloc_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_sorted_bad_alloc(execution_policy(execution::par), IteratorTag());
    test_sorted_bad_alloc(execution_policy(execution::seq), IteratorTag());
    test_sorted_bad_alloc(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_sorted_bad_alloc(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void sorted_bad_alloc_test()
{
    test_sorted_bad_alloc<std::random_access_iterator_tag>();
    test_sorted_bad_alloc<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    sorted_test1();
    sorted_test2();
    sorted_test3();
    sorted_exception_test();
    sorted_bad_alloc_test();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

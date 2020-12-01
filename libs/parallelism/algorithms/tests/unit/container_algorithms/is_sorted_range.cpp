//  Copyright (c) 2015 Daniel Bourgeois
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "is_sorted_range_tests.hpp"

////////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_sorted1()
{
    using namespace hpx::execution;
    test_sorted1(seq, IteratorTag());
    test_sorted1(par, IteratorTag());
    test_sorted1(par_unseq, IteratorTag());

    test_sorted1_async(seq(task), IteratorTag());
    test_sorted1_async(par(task), IteratorTag());

    test_sorted1_seq(IteratorTag());
}

void sorted_test1()
{
    test_sorted1<std::random_access_iterator_tag>();
    test_sorted1<std::forward_iterator_tag>();

    using namespace hpx::execution;

    test_sorted1(seq);
    test_sorted1(par);
    test_sorted1(par_unseq);

    test_sorted1_async(seq(task));
    test_sorted1_async(par(task));

    test_sorted1_seq();
}

////////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_sorted2()
{
    using namespace hpx::execution;
    test_sorted2(seq, IteratorTag());
    test_sorted2(par, IteratorTag());
    test_sorted2(par_unseq, IteratorTag());

    test_sorted2_async(seq(task), IteratorTag());
    test_sorted2_async(par(task), IteratorTag());

    test_sorted2_seq(IteratorTag());
}

void sorted_test2()
{
    test_sorted2<std::random_access_iterator_tag>();
    test_sorted2<std::forward_iterator_tag>();

    using namespace hpx::execution;

    test_sorted2(seq);
    test_sorted2(par);
    test_sorted2(par_unseq);

    test_sorted2_async(seq(task));
    test_sorted2_async(par(task));

    test_sorted2_seq();
}

////////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_sorted3()
{
    using namespace hpx::execution;
    test_sorted3(seq, IteratorTag());
    test_sorted3(par, IteratorTag());
    test_sorted3(par_unseq, IteratorTag());

    test_sorted3_async(seq(task), IteratorTag());
    test_sorted3_async(par(task), IteratorTag());

    test_sorted3_seq(IteratorTag());
}

void sorted_test3()
{
    test_sorted3<std::random_access_iterator_tag>();
    test_sorted3<std::forward_iterator_tag>();

    using namespace hpx::execution;

    test_sorted3(seq);
    test_sorted3(par);
    test_sorted3(par_unseq);

    test_sorted3_async(seq(task));
    test_sorted3_async(par(task));

    test_sorted3_seq();
}

////////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_sorted_exception()
{
    using namespace hpx::execution;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. Therefore we do not test exceptions
    //  with a vector execution policy
    test_sorted_exception(seq, IteratorTag());
    test_sorted_exception(par, IteratorTag());

    test_sorted_exception_async(seq(task), IteratorTag());
    test_sorted_exception_async(par(task), IteratorTag());

    test_sorted_exception_seq(IteratorTag());
}

void sorted_exception_test()
{
    test_sorted_exception<std::random_access_iterator_tag>();
    test_sorted_exception<std::forward_iterator_tag>();

    using namespace hpx::execution;

    test_sorted_exception(seq);
    test_sorted_exception(par);

    test_sorted_exception_async(seq(task));
    test_sorted_exception_async(par(task));

    test_sorted_exception_seq();
}

////////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_sorted_bad_alloc()
{
    using namespace hpx::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_sorted_bad_alloc(par, IteratorTag());
    test_sorted_bad_alloc(seq, IteratorTag());

    test_sorted_bad_alloc_async(seq(task), IteratorTag());
    test_sorted_bad_alloc_async(par(task), IteratorTag());

    test_sorted_bad_alloc_seq(IteratorTag());
}

void sorted_bad_alloc_test()
{
    test_sorted_bad_alloc<std::random_access_iterator_tag>();
    test_sorted_bad_alloc<std::forward_iterator_tag>();

    using namespace hpx::execution;

    test_sorted_bad_alloc(par);
    test_sorted_bad_alloc(seq);

    test_sorted_bad_alloc_async(seq(task));
    test_sorted_bad_alloc_async(par(task));

    test_sorted_bad_alloc_seq();
}

////////////////////////////////////////////////////////////////////////////////
int hpx_main()
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
    using namespace hpx::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

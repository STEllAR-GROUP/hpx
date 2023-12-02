//  Copyright (c) 2015 Daniel Bourgeois
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>

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
int hpx_main()
{
    sorted_test1();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    using namespace hpx::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    hpx::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

//  Copyright (c) 2022 Srinivas Yadav
//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/init.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);

template <typename ExPolicy>
void test_likwid_executor(ExPolicy policy)
{
    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    hpx::for_each(policy.on(hpx::execution::likwid_executor(
                      policy.executor(), "compute")),
        std::begin(c), std::end(c), [](auto t) { return t * t * t; });
}

void test_for_each()
{
    using namespace hpx::execution;

    test_likwid_executor(seq);
    test_likwid_executor(par);
}

int hpx_main(int argc, char* argv[])
{
    test_for_each();
    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

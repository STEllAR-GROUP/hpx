//  Copyright (c) 2022 Srinivas Yadav
//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>
#include <hpx/modules/algorithms.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/executors.hpp>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);

void use_likwid_executor()
{
    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    auto policy = hpx::execution::par;

    hpx::for_each(policy.on(hpx::execution::experimental::likwid_executor(
                      policy.executor(), "compute")),
        std::begin(c), std::end(c), [](auto t) { return t * t * t; });
}

int hpx_main()
{
    use_likwid_executor();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}

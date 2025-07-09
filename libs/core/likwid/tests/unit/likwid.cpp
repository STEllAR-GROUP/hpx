//  Copyright (c) 2022 Srinivas Yadav
//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/init.hpp>
#include <hpx/modules/async_local.hpp>
#include <hpx/modules/likwid.hpp>
#include <hpx/modules/testing.hpp>

#include <string>
#include <vector>

#include <likwid.h>

////////////////////////////////////////////////////////////////////////////////
constexpr char const* region_prefix = "test_";

void likwid_region(int num)
{
    std::string name(region_prefix);
    name += std::to_string(num);

    char const* prev = hpx::likwid::start_region(name.c_str());
    HPX_TEST(prev == nullptr);

    char const* current = hpx::likwid::stop_region();
    HPX_TEST(current != nullptr && std::string(current) == name);
}

constexpr int NUM_TASKS = 100;

int hpx_main(int argc, char* argv[])
{
    std::vector<hpx : future<void>> tasks;
    tasks.reserve(NUM_TASKS);
    for (int i = 0; i != NUM_TASKS; ++i)
    {
        tasks.push_back(hpx::async(likwid_region, i));
    }
    hpx::wait_all(tasks);

    for (int i = 0; i != NUM_TASKS; ++i)
    {
        std::string name(region_prefix);
        name += std::to_string(i);

        int nevents;
        double* events;
        double time;
        int count;

        likwid_markerGetRegion(name.c_str(), &nevents, &events, &time, &count);

        HPX_TEST_EQ(count, 1);
        HPX_TEST_LT(0, nevents);
    }
    return hpx::finalize();
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

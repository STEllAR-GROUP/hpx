//  Copyright (c) 2024 Jiakun Yan
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/modules/testing.hpp>
#include <iostream>
#include <vector>

void run_test();
HPX_PLAIN_ACTION(run_test, run_test_action)

void run_test()
{
    hpx::distributed::barrier::synchronize();
}

int hpx_main(hpx::program_options::variables_map&)
{
    std::vector<hpx::future<void>> futs;
    for (auto l : hpx::find_all_localities())
    {
        futs.emplace_back(hpx::async<run_test_action>(l));
    }
    hpx::wait_all(futs);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    hpx::program_options::options_description description(
        "HPX test barrier synchronize");

    hpx::init_params init_args;
    init_args.desc_cmdline = description;

    HPX_TEST_EQ(hpx::init(argc, argv, init_args), 0);

    std::cout
        << "Runtime shut down. Attempting to call barrier::synchronize()..."
        << std::endl;
    hpx::distributed::barrier::synchronize();
    std::cout << "barrier::synchronize() returned successfully." << std::endl;

    return hpx::util::report_errors();
}

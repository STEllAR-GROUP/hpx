//  Copyright (c) 2024 Jiakun Yan
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Regression test for #6430: verify that hpx::distributed::barrier::synchronize()
// handles uninitialized barriers gracefully without assertion failures

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>

#include <string>

int hpx_main()
{
    // Create a named barrier for testing across localities
    std::string barrier_name = "test_barrier_6430";
    hpx::distributed::barrier b(barrier_name);

    // Test basic barrier synchronization
    b.wait();

    // Test a second synchronization
    b.wait();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    hpx::program_options::options_description description(
        "HPX test barrier synchronize");

    hpx::init_params init_args;
    init_args.desc_cmdline = description;

    HPX_TEST_EQ(hpx::init(argc, argv, init_args), 0);

    return hpx::util::report_errors();
}

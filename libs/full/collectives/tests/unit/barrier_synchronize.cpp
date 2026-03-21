//  Copyright (c) 2024 Jiakun Yan
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Regression test for #6430: verify that hpx::distributed::barrier::synchronize()
// works correctly when called from hpx_main across multiple localities

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>

int hpx_main()
{
    // Test that barrier::synchronize() works from hpx_main
    // This was failing with an assertion before the fix
    hpx::distributed::barrier::synchronize();

    // Do a second synchronization to ensure it works multiple times
    hpx::distributed::barrier::synchronize();

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

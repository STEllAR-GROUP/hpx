//  Copyright (c) 2021 Nanmiao Wu
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char* argv[])
{
    HPX_TEST_EQ(argc, 2);
    HPX_TEST_EQ(std::string(argv[1]), std::string("-wobble=1"));

    return hpx::local::finalize();
}
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // pass unknown command line option that would conflict with predefined
    // alias (-w)
    std::vector<std::string> const cfg = {
        "--hpx:ini=hpx.commandline.allow_unknown!=1",
        "--hpx:ini=hpx.commandline.aliasing!=0"};

    hpx::local::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ(hpx::local::init(hpx_main, argc, argv, init_args), 0);

    return hpx::util::report_errors();
}

//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <string>
#include <vector>

int hpx_main()
{
    HPX_TEST_EQ(hpx::get_config_entry("hpx.cores", "1"), std::string("3"));
    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {"--hpx:cores=3"};

    hpx::local::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ(hpx::local::init(hpx_main, argc, argv, init_args), 0);

    return hpx::util::report_errors();
}

//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/modules/testing.hpp>

int hpx_main()
{
    HPX_TEST_EQ(hpx::get_config_entry("hpx.cores", "1"), std::string("3"));
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg =
    {
        "--hpx:cores=3"
    };

    hpx::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ(hpx::init(argc, argv, init_args), 0);

    return hpx::util::report_errors();
}

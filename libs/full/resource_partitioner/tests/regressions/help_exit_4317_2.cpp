//  Copyright (c) 2020 albestro
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <string>
#include <vector>

std::atomic<bool> main_executed(false);

int hpx_main()
{
    main_executed = true;
    return hpx::finalize();
}

int main(int argc, char** argv)
{
    std::vector<std::string> cfg = {"--hpx:help"};

    hpx::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ(hpx::init(argc, argv, init_args), 0);

    HPX_TEST(!main_executed);

    return hpx::util::report_errors();
}

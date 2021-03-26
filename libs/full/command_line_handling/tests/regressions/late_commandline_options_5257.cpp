//  Copyright (c) 2021 Hartmut Kaiser
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

std::atomic<bool> invoked(false);

int hpx_main(int, char*[])
{
    invoked = true;
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    hpx::init_params params;
    params.cfg = { "--hpx:exit" };

    HPX_TEST_EQ(hpx::init(argc, argv, params), 1);
    HPX_TEST(!invoked);

    return hpx::util::report_errors();
}

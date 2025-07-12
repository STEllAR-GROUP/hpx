//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {"hpx.os_threads=4"};

    hpx::init_params init_args;
    init_args.cfg = cfg;

    hpx::start(argc, argv, init_args);

    std::atomic_bool invoked(false);

    std::vector<int> vs(65536);
    hpx::for_each(hpx::execution::par, vs.begin(), vs.end(),
        [&](int) { invoked = true; });

    hpx::stop();

    HPX_TEST(invoked.load());

    return hpx::util::report_errors();
}

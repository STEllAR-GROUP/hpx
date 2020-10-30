//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <chrono>

// This test ensures that thread creation uses the correct stack sizes. We
// slightly change all the stack sizes in the configuration to catch problems
// with the used stack sizes not matching the configured sizes.

int hpx_main()
{
    hpx::this_thread::sleep_for(std::chrono::milliseconds(10));

    hpx::thread t([]() {});
    t.join();

    return hpx::finalize();
}

int main(int argc, char** argv)
{
    hpx::init_params p;
    p.cfg = {"hpx.stacks.small_size=" +
            std::to_string(HPX_SMALL_STACK_SIZE + 0x1000),
        "hpx.stacks.medium_size=" +
            std::to_string(HPX_MEDIUM_STACK_SIZE + 0x1000),
        "hpx.stacks.large_size=" +
            std::to_string(HPX_LARGE_STACK_SIZE + 0x1000),
        "hpx.stacks.huge_size=" + std::to_string(HPX_HUGE_STACK_SIZE + 0x1000)};

    return hpx::init(argc, argv, p);
}

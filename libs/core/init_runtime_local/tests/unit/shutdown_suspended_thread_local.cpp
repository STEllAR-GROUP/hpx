//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This test checks that the runtime takes into account suspended threads before
// initiating full shutdown.

#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <chrono>

int hpx_main()
{
    hpx::post(
        [] { hpx::this_thread::sleep_for(std::chrono::milliseconds(500)); });

    return hpx::local::finalize();
}

int main(int argc, char** argv)
{
    HPX_TEST_EQ(hpx::local::init(hpx_main, argc, argv), 0);

    return hpx::util::report_errors();
}

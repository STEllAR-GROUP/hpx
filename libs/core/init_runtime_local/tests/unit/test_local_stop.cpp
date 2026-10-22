//  Copyright (c) 2026 Shivansh Singh
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

int hpx_main()
{
    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    // Test 1: Start and stop the runtime
    HPX_TEST(hpx::local::start(hpx_main, argc, argv));
    HPX_TEST_EQ(hpx::local::stop(), 0);

    // Test 2: Calling stop() again when the runtime has already stopped must
    // not cause a double-free or dereference a dangling pointer.
    {
        hpx::error_code ec(hpx::throwmode::lightweight);
        int const res = hpx::local::stop(ec);
        HPX_TEST(ec);
        HPX_TEST_EQ(ec.value(), hpx::error::invalid_status);
        HPX_TEST_EQ(res, -1);
    }

    // Test 3: Multiple sequential start/stop lifecycles must succeed cleanly
    for (int i = 0; i < 3; ++i)
    {
        HPX_TEST(hpx::local::start(hpx_main, argc, argv));
        HPX_TEST_EQ(hpx::local::stop(), 0);

        hpx::error_code ec(hpx::throwmode::lightweight);
        int const res = hpx::local::stop(ec);
        HPX_TEST(ec);
        HPX_TEST_EQ(ec.value(), hpx::error::invalid_status);
        HPX_TEST_EQ(res, -1);
    }

    return hpx::util::report_errors();
}

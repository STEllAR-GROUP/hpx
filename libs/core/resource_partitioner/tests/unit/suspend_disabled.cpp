//  Copyright (c) 2017 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/init.hpp>
#include <hpx/modules/resource_partitioner.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/thread_pool_util.hpp>
#include <hpx/thread.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

int hpx_main()
{
    // Try suspending without elasticity enabled, should throw an exception
    bool exception_thrown = false;

    try
    {
        hpx::threads::thread_pool_base& tp =
            hpx::resource::get_thread_pool("default");

        // Use .get() to throw exception
        hpx::threads::suspend_processing_unit(tp, 0).get();
        HPX_TEST_MSG(false,
            "Suspending should not be allowed with "
            "elasticity disabled");
    }
    catch (hpx::exception const&)
    {
        exception_thrown = true;
    }

    HPX_TEST(exception_thrown);

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    hpx::local::init_params init_args;

    init_args.cfg = {"hpx.os_threads=" +
        std::to_string(((std::min) (std::size_t(4),
            std::size_t(hpx::threads::hardware_concurrency()))))};
    init_args.rp_callback = [](auto& rp,
                                hpx::program_options::variables_map const&) {
        // Explicitly disable elasticity if it is in defaults
        rp.create_thread_pool("default",
            hpx::resource::scheduling_policy::local_priority_fifo,
            hpx::threads::policies::scheduler_mode(
                hpx::threads::policies::scheduler_mode::default_ &
                ~hpx::threads::policies::scheduler_mode::enable_elasticity));
    };

    HPX_TEST_EQ(hpx::local::init(hpx_main, argc, argv, init_args), 0);
}

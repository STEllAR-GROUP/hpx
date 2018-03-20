//  Copyright (c) 2017 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/hpx_init.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

int hpx_main(int argc, char* argv[])
{
    // Try suspending without elasticity enabled, should throw an exception
    bool exception_thrown = false;

    try
    {
        hpx::threads::thread_pool_base& tp =
            hpx::resource::get_thread_pool("default");

        // Use .get() to throw exception
        tp.suspend_processing_unit(0).get();
        HPX_TEST_MSG(false, "Suspending should not be allowed with "
            "elasticity disabled");
    }
    catch (hpx::exception const&)
    {
        exception_thrown = true;
    }

    HPX_TEST(exception_thrown);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg =
    {
        "hpx.os_threads=4"
    };

    hpx::resource::partitioner rp(argc, argv, std::move(cfg));
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
}

//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/hpx_init.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/testing.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

int hpx_main(int argc, char* argv[])
{
    std::size_t num_threads = hpx::resource::get_num_threads("default");
    hpx::threads::thread_pool_base& tp =
                hpx::resource::get_thread_pool("default");

    auto used_pu_mask = tp.get_used_processing_units();
    HPX_TEST_EQ(hpx::threads::count(used_pu_mask), num_threads);

    for (std::size_t t = 0; t < num_threads; ++t)
    {
        auto thread_mask = hpx::resource::get_partitioner().get_pu_mask(t);
        HPX_TEST(hpx::threads::bit_or(used_pu_mask, thread_mask));
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg = {
        "hpx.os_threads=4"
    };

    // set up the resource partitioner
    hpx::resource::partitioner rp(argc, argv, std::move(cfg));

    // now run the test
    HPX_TEST_EQ(hpx::init(), 0);
    return hpx::util::report_errors();
}

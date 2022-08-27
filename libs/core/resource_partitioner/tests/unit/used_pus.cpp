//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/local/init.hpp>
#include <hpx/local/thread.hpp>
#include <hpx/modules/resource_partitioner.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

int hpx_main()
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

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    hpx::local::init_params init_args;
    init_args.cfg = {"hpx.os_threads=" +
        std::to_string(((std::min)(std::size_t(4),
            std::size_t(hpx::threads::hardware_concurrency()))))};

    // now run the test
    HPX_TEST_EQ(hpx::local::init(hpx_main, argc, argv, init_args), 0);
    return hpx::util::report_errors();
}

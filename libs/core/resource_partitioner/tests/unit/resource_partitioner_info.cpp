//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/assert.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/resource_partitioner.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

std::size_t const max_threads = (std::min) (std::size_t(4),
    std::size_t(hpx::threads::hardware_concurrency()));

int hpx_main()
{
    std::size_t const num_os_threads = hpx::get_os_thread_count();
    HPX_TEST_EQ(num_os_threads, hpx::resource::get_num_threads());
    HPX_TEST_EQ(num_os_threads, hpx::resource::get_num_threads(0));
    HPX_TEST_EQ(std::size_t(1), hpx::resource::get_num_thread_pools());
    HPX_TEST_EQ(std::size_t(0), hpx::resource::get_pool_index("default"));
    HPX_TEST_EQ(std::string("default"), hpx::resource::get_pool_name(0));

    {
        hpx::threads::thread_pool_base& pool =
            hpx::resource::get_thread_pool(0);
        HPX_TEST_EQ(std::size_t(0), pool.get_pool_index());
        HPX_TEST_EQ(std::string("default"), pool.get_pool_name());
        HPX_TEST_EQ(std::size_t(0), pool.get_thread_offset());
    }

    {
        hpx::threads::thread_pool_base& pool =
            hpx::resource::get_thread_pool("default");
        HPX_TEST_EQ(std::size_t(0), pool.get_pool_index());
        HPX_TEST_EQ(std::string("default"), pool.get_pool_name());
        HPX_TEST_EQ(std::size_t(0), pool.get_thread_offset());
    }

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_ASSERT(max_threads >= 2);

    hpx::local::init_params init_args;
    init_args.cfg = {"hpx.os_threads=" +
        std::to_string(((std::min) (std::size_t(4),
            std::size_t(hpx::threads::hardware_concurrency()))))};

    // now run the test
    HPX_TEST_EQ(hpx::local::init(hpx_main, argc, argv, init_args), 0);
    return hpx::util::report_errors();
}

//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/hpx_init.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

int hpx_main()
{
    HPX_TEST_EQ(std::size_t(4), hpx::resource::get_num_threads());
    HPX_TEST_EQ(std::size_t(4), hpx::resource::get_num_threads(0));
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

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg = {"hpx.os_threads=4"};

    hpx::init_params init_args;
    init_args.cfg = std::move(cfg);

    // now run the test
    HPX_TEST_EQ(hpx::init(argc, argv, init_args), 0);
    return hpx::util::report_errors();
}

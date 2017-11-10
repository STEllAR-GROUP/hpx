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
#include <string>
#include <utility>
#include <vector>

int hpx_main(int argc, char* argv[])
{
    std::size_t const num_threads = hpx::resource::get_num_threads("default");

    HPX_TEST_EQ(std::size_t(4), num_threads);

    hpx::threads::detail::thread_pool_base& tp =
        hpx::resource::get_thread_pool("default");

    HPX_TEST_EQ(hpx::threads::count(tp.get_used_processing_units()), std::size_t(4));

    // Remove all but one pu
    for (std::size_t thread_num = 0; thread_num < num_threads - 1; ++thread_num)
    {
        tp.remove_processing_unit(thread_num);
    }

    // Schedule lots of dummy work
    for (std::size_t i = 0;
         i < hpx::resource::get_num_threads("default") * 10000;
         ++i)
    {
        hpx::async([](){});
    }

    // Start shutdown
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

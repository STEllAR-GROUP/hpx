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
        tp.suspend_processing_unit(thread_num);
    }

    // Schedule some dummy work
    for (std::size_t i = 0; i < 100000; ++i)
    {
        hpx::async([](){});
    }

    // Start shutdown
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg =
    {
        "hpx.os_threads=4"
    };

    using hpx::resource::scheduling_policy;

    // TODO: Loop through integer values?
    // TODO: Should all schedulers support suspension?
    std::vector<scheduling_policy> const policies =
    {
        //scheduling_policy::local,
        scheduling_policy::local_priority_fifo,
        scheduling_policy::local_priority_lifo,
        //scheduling_policy::static_,
        //scheduling_policy::static_priority,
        //scheduling_policy::abp_priority,
        //scheduling_policy::hierarchy,
        //scheduling_policy::periodic_priority,
        //scheduling_policy::throttle
    };

    for (auto policy : policies)
    {
        // Set up the resource partitioner
        hpx::resource::partitioner rp(argc, argv, std::move(cfg));
        rp.create_thread_pool("default", policy);

        HPX_TEST_EQ(hpx::init(argc, argv), 0);
    }

    return hpx::util::report_errors();
}

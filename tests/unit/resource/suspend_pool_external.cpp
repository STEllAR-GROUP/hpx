//  Copyright (c) 2018 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/compat/thread.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/threadmanager.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/policies/schedulers.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

int hpx_main(int argc, char* argv[])
{
    return 0;
}

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg =
    {
        "hpx.os_threads=4"
    };

    hpx::start(argc, argv, std::move(cfg));

    hpx::threads::executors::pool_executor default_exec("default");
    hpx::threads::thread_pool_base& default_pool =
        hpx::resource::get_thread_pool("default");
    std::size_t const default_pool_threads =
        hpx::resource::get_num_threads("default");

    hpx::util::high_resolution_timer t;

    while (t.elapsed() < 5)
    {
        for (std::size_t i = 0;
             i < default_pool_threads * 10000; ++i)
        {
            hpx::apply(default_exec, [](){});
        }

        bool suspended = false;
        default_pool.suspend_cb([&suspended]()
            {
                suspended = true;
            });

        while (!suspended)
        {
            hpx::compat::this_thread::yield();
        }

        bool resumed = false;
        default_pool.resume_cb([&resumed]()
            {
                resumed = true;
            });

        while (!resumed)
        {
            hpx::compat::this_thread::yield();
        }
    }

    hpx::apply(default_exec, []() { hpx::finalize(); });

    HPX_TEST_EQ(hpx::stop(), 0);

    return hpx::util::report_errors();
}

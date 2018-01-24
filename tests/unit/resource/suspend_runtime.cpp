//  Copyright (c) 2017 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/compat/thread.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/hpx_suspend.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threadmanager.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/policies/schedulers.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

int hpx_main(int, char**)
{
    return 0;
}

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg =
    {
        "hpx.os_threads=4"
    };


    hpx::start(argc, argv);
    hpx::suspend();

    for (std::size_t i = 0; i < 100; ++i)
    {
        hpx::resume();

        hpx::async([]()
            {
                for (std::size_t i = 0; i < 10000; ++i)
                {
                    hpx::async([](){});
                }
            });

        hpx::suspend();
    }

    hpx::resume();
    hpx::async([]() { hpx::finalize(); });
    hpx::stop();

    return hpx::util::report_errors();
}

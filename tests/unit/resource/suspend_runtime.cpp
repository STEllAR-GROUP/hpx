//  Copyright (c) 2017 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/hpx_start.hpp>
#include <hpx/hpx_suspend.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/threadmanager.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/yield_while.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

int main(int argc, char* argv[])
{
    std::vector<std::string> scheduler_strings =
    {
        "local",
        "local-priority-lifo",
        "local-priority-fifo",
        "static",
        "static-priority"
    };

    for (auto const& scheduler_string : scheduler_strings)
    {
        std::vector<std::string> cfg =
            {
                "hpx.os_threads=4",
                "hpx.scheduler=" + scheduler_string
            };

        hpx::start(nullptr, argc, argv, cfg);

        // Wait for runtime to start
        hpx::runtime* rt = hpx::get_runtime_ptr();
        hpx::util::yield_while([rt]()
            { return rt->get_state() < hpx::state_running; });

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
    }

    return hpx::util::report_errors();
}

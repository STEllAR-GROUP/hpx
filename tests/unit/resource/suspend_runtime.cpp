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
    std::cout << "hello from hpx_main!\n";
    return 0;
}

int main(int argc, char* argv[])
{
    // std::vector<std::string> cfg =
    // {
    //     "hpx.os_threads=4"
    // };

    // hpx::resource::partitioner rp(argc, argv); //, std::move(cfg));


    {
        // hpx::start(hpx::util::function_nonser<int(int, char**)>(), argc, argv);
        hpx::start(argc, argv);

        for (std::size_t i = 0; i < 10; ++i)
        {
            double x = 13.2;
            hpx::async([&x]()
                {
                    for (std::size_t i = 0; i < 20000; ++i)
                    {
                        hpx::async([&x]()
                            {
                                for (std::size_t j = 0; j < 20000; ++j)
                                {
                                    x -= sin(17 * x - 3.2);
                                }
                            });
                    }
                });

            hpx::suspend();
            std::cout << x << "\n";
            hpx::compat::this_thread::sleep_for(std::chrono::seconds(4));
            hpx::resume();
        }

        hpx::async([]() { hpx::finalize(); });
        hpx::stop();
    }

    return hpx::util::report_errors();
}

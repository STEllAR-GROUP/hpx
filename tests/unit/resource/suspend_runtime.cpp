//  Copyright (c) 2017 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/hpx_start.hpp>
#include <hpx/hpx_suspend.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/threadmanager.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/testing.hpp>
#include <hpx/util/yield_while.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

void test_scheduler(int argc, char* argv[],
    hpx::resource::scheduling_policy scheduler)
{
    std::vector<std::string> cfg =
    {
        "hpx.os_threads=4"
    };

    hpx::resource::partitioner rp(nullptr, argc, argv, std::move(cfg));

    rp.create_thread_pool("default", scheduler);

    hpx::start(nullptr, argc, argv);

    // Wait for runtime to start
    hpx::runtime* rt = hpx::get_runtime_ptr();
    hpx::util::yield_while([rt]()
        { return rt->get_state() < hpx::state_running; });

    hpx::suspend();

    for (std::size_t i = 0; i < 100; ++i)
    {
        hpx::resume();

        hpx::apply([]()
            {
                for (std::size_t i = 0; i < 10000; ++i)
                {
                    hpx::apply([](){});
                }
            });

        hpx::suspend();
    }

    hpx::resume();
    hpx::apply([]() { hpx::finalize(); });
    HPX_TEST_EQ(hpx::stop(), 0);
}

int main(int argc, char* argv[])
{
    std::vector<hpx::resource::scheduling_policy> schedulers = {
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
        hpx::resource::scheduling_policy::local,
        hpx::resource::scheduling_policy::local_priority_fifo,
#if defined(HPX_HAVE_CXX11_STD_ATOMIC_128BIT)
        hpx::resource::scheduling_policy::local_priority_lifo,
#endif
#endif
#if defined(HPX_HAVE_ABP_SCHEDULER) && defined(HPX_HAVE_CXX11_STD_ATOMIC_128BIT)
        hpx::resource::scheduling_policy::abp_priority_fifo,
        hpx::resource::scheduling_policy::abp_priority_lifo,
#endif
#if defined(HPX_HAVE_STATIC_SCHEDULER)
        hpx::resource::scheduling_policy::static_,
#endif
#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
        hpx::resource::scheduling_policy::static_priority,
#endif
#if defined(HPX_HAVE_SHARED_PRIORITY_SCHEDULER)
        hpx::resource::scheduling_policy::shared_priority,
#endif
    };

    for (auto const scheduler : schedulers)
    {
        test_scheduler(argc, argv, scheduler);
    }

    return hpx::util::report_errors();
}

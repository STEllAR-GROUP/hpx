//  Copyright (c) 2017 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/execution_base/this_thread.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/hpx_suspend.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/include/threadmanager.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/timing.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

void test_scheduler(
    int argc, char* argv[], hpx::resource::scheduling_policy scheduler)
{
    hpx::init_params init_args;

    init_args.cfg = {"hpx.os_threads=4"};
    init_args.rp_callback = [scheduler](auto& rp,
                                hpx::program_options::variables_map const&) {
        rp.create_thread_pool("default", scheduler);
    };

    hpx::start(nullptr, argc, argv, init_args);
    hpx::suspend();

    hpx::chrono::high_resolution_timer t;

    while (t.elapsed() < 2)
    {
        hpx::resume();

        hpx::apply([]() {
            for (std::size_t i = 0; i < 10000; ++i)
            {
                hpx::apply([]() {});
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

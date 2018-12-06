//  Copyright (c) 2017 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/hpx_init.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/policies/schedulers.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

int hpx_main(int argc, char* argv[])
{
    std::size_t const num_threads = hpx::resource::get_num_threads("default");

    HPX_TEST_EQ(std::size_t(4), num_threads);

    hpx::threads::thread_pool_base& tp =
        hpx::resource::get_thread_pool("default");

    HPX_TEST_EQ(tp.get_active_os_thread_count(), std::size_t(4));

    // Remove all but one pu
    for (std::size_t thread_num = 0; thread_num < num_threads - 1; ++thread_num)
    {
        tp.suspend_processing_unit(thread_num).get();
    }

    // Schedule some dummy work
    for (std::size_t i = 0; i < 100000; ++i)
    {
        hpx::apply([](){});
    }

    // Start shutdown
    return hpx::finalize();
}

void test_scheduler(int argc, char* argv[],
    hpx::resource::scheduling_policy scheduler)
{
    std::vector<std::string> cfg =
    {
        "hpx.os_threads=4"
    };

    hpx::resource::partitioner rp(argc, argv, std::move(cfg));

    rp.create_thread_pool("default", scheduler,
        hpx::threads::policies::scheduler_mode(
            hpx::threads::policies::default_mode |
            hpx::threads::policies::enable_elasticity));

    HPX_TEST_EQ(hpx::init(argc, argv), 0);
}

int main(int argc, char* argv[])
{
    // NOTE: Periodic priority scheduler not tested because it does not take
    // into account scheduler states when scheduling work.

    {
        // These schedulers should succeed
        std::vector<hpx::resource::scheduling_policy> schedulers =
            {
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
                hpx::resource::scheduling_policy::local,
                hpx::resource::scheduling_policy::local_priority_fifo,
                hpx::resource::scheduling_policy::local_priority_lifo,
#endif
#if defined(HPX_HAVE_ABP_SCHEDULER)
                hpx::resource::scheduling_policy::abp_priority_fifo,
                hpx::resource::scheduling_policy::abp_priority_lifo,
#endif
#if defined(HPX_HAVE_SHARED_PRIORITY_SCHEDULER)
                hpx::resource::scheduling_policy::shared_priority,
#endif
            };

        for (auto const scheduler : schedulers)
        {
            test_scheduler(argc, argv, scheduler);
        }
    }

    {
        // These schedulers should fail
        std::vector<hpx::resource::scheduling_policy> schedulers =
        {
#if defined(HPX_HAVE_STATIC_SCHEDULER)
            hpx::resource::scheduling_policy::static_,
#endif
#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
            hpx::resource::scheduling_policy::static_priority,
#endif
        };

        for (auto const scheduler : schedulers)
        {
            bool exception_thrown = false;
            try
            {
                test_scheduler(argc, argv, scheduler);
            }
            catch (hpx::exception const&)
            {
                exception_thrown = true;
            }

            HPX_TEST(exception_thrown);
        }
    }

    return hpx::util::report_errors();
}

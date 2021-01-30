//  Copyright (c) 2018 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/modules/schedulers.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/threading_base/scheduler_mode.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

int hpx_main()
{
    bool run = false;
    hpx::future<void> f1 = hpx::async([&run]() { run = true; });

    if (!run)
    {
        // This thread should get scheduled last (because of
        // hpx::threads::thread_schedule_state::pending) and let the function
        // spawned above run.
        hpx::this_thread::suspend(hpx::threads::thread_schedule_state::pending);
    }

    HPX_TEST(run);

    return hpx::finalize();
}

template <typename Scheduler>
void test_scheduler(int argc, char* argv[])
{
    hpx::init_params init_args;

    init_args.cfg = {"hpx.os_threads=1"};
    init_args.rp_callback = [](auto& rp,
                                hpx::program_options::variables_map const&) {
        rp.create_thread_pool("default",
            [](hpx::threads::thread_pool_init_parameters thread_pool_init,
                hpx::threads::policies::thread_queue_init_parameters
                    thread_queue_init)
                -> std::unique_ptr<hpx::threads::thread_pool_base> {
                typename Scheduler::init_parameter_type init(
                    thread_pool_init.num_threads_,
                    thread_pool_init.affinity_data_, std::size_t(-1),
                    thread_queue_init);
                std::unique_ptr<Scheduler> scheduler(new Scheduler(init));

                thread_pool_init.mode_ = hpx::threads::policies::scheduler_mode(
                    hpx::threads::policies::do_background_work |
                    hpx::threads::policies::reduce_thread_priority |
                    hpx::threads::policies::delay_exit);

                std::unique_ptr<hpx::threads::thread_pool_base> pool(
                    new hpx::threads::detail::scheduled_thread_pool<Scheduler>(
                        std::move(scheduler), thread_pool_init));

                return pool;
            });
    };

    HPX_TEST_EQ(hpx::init(argc, argv, init_args), 0);
}

int main(int argc, char* argv[])
{
#if defined(HPX_HAVE_CXX11_STD_ATOMIC_128BIT)
    {
        using scheduler_type =
            hpx::threads::policies::local_priority_queue_scheduler<std::mutex,
                hpx::threads::policies::lockfree_lifo>;
        test_scheduler<scheduler_type>(argc, argv);
    }
#endif

    {
        using scheduler_type =
            hpx::threads::policies::local_priority_queue_scheduler<std::mutex,
                hpx::threads::policies::lockfree_fifo>;
        test_scheduler<scheduler_type>(argc, argv);
    }

#if defined(HPX_HAVE_ABP_SCHEDULER) && defined(HPX_HAVE_CXX11_STD_ATOMIC_128BIT)
    {
        using scheduler_type =
            hpx::threads::policies::local_priority_queue_scheduler<std::mutex,
                hpx::threads::policies::lockfree_abp_lifo>;
        test_scheduler<scheduler_type>(argc, argv);
    }

    {
        using scheduler_type =
            hpx::threads::policies::local_priority_queue_scheduler<std::mutex,
                hpx::threads::policies::lockfree_abp_fifo>;
        test_scheduler<scheduler_type>(argc, argv);
    }
#endif

    return hpx::util::report_errors();
}

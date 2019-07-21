//  Copyright (c) 2018 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/policies/schedulers.hpp>
#include <hpx/testing.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

int hpx_main(int argc, char* argv[])
{
    bool run = false;
    hpx::future<void> f1 = hpx::async([&run]()
        {
            run = true;
        });

    if (!run)
    {
        // This thread should get scheduled last (because of
        // hpx::threads::pending) and let the function spawned above run.
        hpx::this_thread::suspend(hpx::threads::pending);
    }

    HPX_TEST(run);

    return hpx::finalize();
}

template <typename Scheduler>
void test_scheduler(int argc, char* argv[])
{
    std::vector<std::string> cfg =
    {
        "hpx.os_threads=1"
    };

    hpx::resource::partitioner rp(argc, argv, std::move(cfg));

    rp.create_thread_pool("default",
        [](hpx::threads::policies::callback_notifier& notifier,
            std::size_t num_threads, std::size_t thread_offset,
            std::size_t pool_index, std::string const& pool_name)
        -> std::unique_ptr<hpx::threads::thread_pool_base>
        {
            typename Scheduler::init_parameter_type init(num_threads);
            std::unique_ptr<Scheduler> scheduler(new Scheduler(init));

            auto mode = hpx::threads::policies::scheduler_mode(
                hpx::threads::policies::do_background_work |
                hpx::threads::policies::reduce_thread_priority |
                hpx::threads::policies::delay_exit);

            std::unique_ptr<hpx::threads::thread_pool_base> pool(
                new hpx::threads::detail::scheduled_thread_pool<Scheduler>(
                    std::move(scheduler), notifier, pool_index, pool_name, mode,
                    thread_offset));

            return pool;
        });

    HPX_TEST_EQ(hpx::init(argc, argv), 0);
}

int main(int argc, char* argv[])
{
    {
        using scheduler_type =
            hpx::threads::policies::local_priority_queue_scheduler<
                std::mutex, hpx::threads::policies::lockfree_lifo
            >;
        test_scheduler<scheduler_type>(argc, argv);
    }

    {
        using scheduler_type =
            hpx::threads::policies::local_priority_queue_scheduler<
                std::mutex, hpx::threads::policies::lockfree_fifo
            >;
        test_scheduler<scheduler_type>(argc, argv);
    }

#if defined(HPX_HAVE_ABP_SCHEDULER) && defined(HPX_HAVE_CXX11_STD_ATOMIC_128BIT)
    {
        using scheduler_type =
            hpx::threads::policies::local_priority_queue_scheduler<
                std::mutex, hpx::threads::policies::lockfree_abp_lifo
            >;
        test_scheduler<scheduler_type>(argc, argv);
    }

    {
        using scheduler_type =
            hpx::threads::policies::local_priority_queue_scheduler<
                std::mutex, hpx::threads::policies::lockfree_abp_fifo
            >;
        test_scheduler<scheduler_type>(argc, argv);
    }
#endif

    return hpx::util::report_errors();
}

//  Copyright (c) 2017 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/modules/schedulers.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/timing.hpp>
#include <hpx/threading_base/scheduler_mode.hpp>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

int hpx_main()
{
    std::size_t const num_threads = hpx::resource::get_num_threads("default");

    HPX_TEST_EQ(std::size_t(4), num_threads);

    hpx::threads::thread_pool_base& tp =
        hpx::resource::get_thread_pool("default");

    HPX_TEST_EQ(tp.get_active_os_thread_count(), std::size_t(4));

    {
        // Check number of used resources
        for (std::size_t thread_num = 0; thread_num < num_threads - 1;
             ++thread_num)
        {
            hpx::threads::suspend_processing_unit(tp, thread_num).get();
            HPX_TEST_EQ(std::size_t(num_threads - thread_num - 1),
                tp.get_active_os_thread_count());
        }

        for (std::size_t thread_num = 0; thread_num < num_threads - 1;
             ++thread_num)
        {
            hpx::threads::resume_processing_unit(tp, thread_num).get();
            HPX_TEST_EQ(
                std::size_t(thread_num + 2), tp.get_active_os_thread_count());
        }
    }

    {
        // Check suspending pu on which current thread is running.

        // NOTE: This only works as long as there is another OS thread which has
        // no work and is able to steal.
        std::size_t worker_thread_num = hpx::get_worker_thread_num();
        hpx::threads::suspend_processing_unit(tp, worker_thread_num).get();
        hpx::threads::resume_processing_unit(tp, worker_thread_num).get();
    }

    {
        // Check when suspending all but one, we end up on the same thread
        std::size_t thread_num = 0;
        auto test_function = [&thread_num, &tp]() {
            HPX_TEST_EQ(thread_num + tp.get_thread_offset(),
                hpx::get_worker_thread_num());
        };

        for (thread_num = 0; thread_num < num_threads; ++thread_num)
        {
            for (std::size_t thread_num_suspend = 0;
                 thread_num_suspend < num_threads; ++thread_num_suspend)
            {
                if (thread_num != thread_num_suspend)
                {
                    hpx::threads::suspend_processing_unit(
                        tp, thread_num_suspend)
                        .get();
                }
            }

            hpx::async(test_function).wait();

            for (std::size_t thread_num_resume = 0;
                 thread_num_resume < num_threads; ++thread_num_resume)
            {
                if (thread_num != thread_num_resume)
                {
                    hpx::threads::resume_processing_unit(tp, thread_num_resume)
                        .get();
                }
            }
        }
    }

    {
        // Check suspending and resuming the same thread without waiting for
        // each to finish.
        for (std::size_t thread_num = 0;
             thread_num < hpx::resource::get_num_threads("default");
             ++thread_num)
        {
            std::vector<hpx::future<void>> fs;

            fs.push_back(hpx::threads::suspend_processing_unit(tp, thread_num));
            fs.push_back(hpx::threads::resume_processing_unit(tp, thread_num));

            hpx::wait_all(fs);

            // Suspend is not guaranteed to run before resume, so make sure
            // processing unit is running
            hpx::threads::resume_processing_unit(tp, thread_num).get();

            fs.clear();

            // Launching 4 (i.e. same as number of threads) tasks may deadlock
            // as no thread is available to steal from the current thread.
            fs.push_back(hpx::threads::suspend_processing_unit(tp, thread_num));
            fs.push_back(hpx::threads::suspend_processing_unit(tp, thread_num));
            fs.push_back(hpx::threads::suspend_processing_unit(tp, thread_num));

            hpx::wait_all(fs);

            fs.clear();

            // Launching 4 (i.e. same as number of threads) tasks may deadlock
            // as no thread is available to steal from the current thread.
            fs.push_back(hpx::threads::resume_processing_unit(tp, thread_num));
            fs.push_back(hpx::threads::resume_processing_unit(tp, thread_num));
            fs.push_back(hpx::threads::resume_processing_unit(tp, thread_num));

            hpx::wait_all(fs);
        }
    }

    {
        // Check random scheduling with reducing resources.
        std::size_t thread_num = 0;
        bool up = true;
        std::vector<hpx::future<void>> fs;
        hpx::chrono::high_resolution_timer t;
        while (t.elapsed() < 2)
        {
            for (std::size_t i = 0;
                 i < hpx::resource::get_num_threads("default") * 10; ++i)
            {
                fs.push_back(hpx::async([]() {}));
            }

            if (up)
            {
                if (thread_num != hpx::resource::get_num_threads("default") - 1)
                {
                    hpx::threads::suspend_processing_unit(tp, thread_num).get();
                }

                ++thread_num;

                if (thread_num == hpx::resource::get_num_threads("default"))
                {
                    up = false;
                    --thread_num;
                }
            }
            else
            {
                hpx::threads::resume_processing_unit(tp, thread_num - 1).get();

                --thread_num;

                if (thread_num == 0)
                {
                    up = true;
                }
            }
        }

        hpx::when_all(std::move(fs)).get();

        // Don't exit with suspended pus
        for (std::size_t thread_num_resume = 0; thread_num_resume < thread_num;
             ++thread_num_resume)
        {
            hpx::threads::resume_processing_unit(tp, thread_num_resume).get();
        }
    }

    return hpx::finalize();
}

void test_scheduler(
    int argc, char* argv[], hpx::resource::scheduling_policy scheduler)
{
    hpx::init_params init_args;
    init_args.cfg = {"hpx.os_threads=4"};
    init_args.rp_callback = [scheduler](auto& rp,
                                hpx::program_options::variables_map const&) {
        rp.create_thread_pool("default", scheduler,
            hpx::threads::policies::scheduler_mode(
                hpx::threads::policies::default_mode |
                hpx::threads::policies::enable_elasticity));
    };

    HPX_TEST_EQ(hpx::init(argc, argv, init_args), 0);
}

int main(int argc, char* argv[])
{
    // NOTE: Static schedulers do not support suspending the own worker thread
    // because they do not steal work.

    {
        // These schedulers should succeed
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
        std::vector<hpx::resource::scheduling_policy> schedulers = {
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

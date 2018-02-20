//  Copyright (c) 2017 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/policies/schedulers.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <memory>
#include <stdexcept>
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

    {
        // Check number of used resources
        for (std::size_t thread_num = 0; thread_num < num_threads - 1; ++thread_num)
        {
            tp.suspend_processing_unit(thread_num).get();
            HPX_TEST_EQ(std::size_t(num_threads - thread_num - 1),
                tp.get_active_os_thread_count());
        }

        for (std::size_t thread_num = 0; thread_num < num_threads - 1; ++thread_num)
        {
            tp.resume_processing_unit(thread_num).get();
            HPX_TEST_EQ(std::size_t(thread_num + 2),
                tp.get_active_os_thread_count());
        }
    }

    {
        // Check suspending pu on which current thread is running.

        // NOTE: This only works as long as there is another OS thread which has
        // no work and is able to steal.
        std::size_t worker_thread_num = hpx::get_worker_thread_num();
        tp.suspend_processing_unit(worker_thread_num).get();
        tp.resume_processing_unit(worker_thread_num).get();
    }

    {
        // Check when suspending all but one, we end up on the same thread
        std::size_t thread_num = 0;
        auto test_function = [&thread_num, &tp]()
        {
            HPX_TEST_EQ(thread_num + tp.get_thread_offset(),
                hpx::get_worker_thread_num());
        };

        for (thread_num = 0; thread_num < num_threads;
            ++thread_num)
        {
            for (std::size_t thread_num_suspend = 0;
                thread_num_suspend < num_threads;
                ++thread_num_suspend)
            {
                if (thread_num != thread_num_suspend)
                {
                    tp.suspend_processing_unit(thread_num_suspend).get();
                }
            }

            hpx::async(test_function).get();

            for (std::size_t thread_num_resume = 0;
                thread_num_resume < num_threads;
                ++thread_num_resume)
            {
                if (thread_num != thread_num_resume)
                {
                    tp.resume_processing_unit(thread_num_resume).get();
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

            fs.push_back(tp.suspend_processing_unit(thread_num));
            fs.push_back(tp.resume_processing_unit(thread_num));

            hpx::wait_all(fs);

            // Suspend is not guaranteed to run before resume, so make sure
            // processing unit is running
            tp.resume_processing_unit(thread_num).get();

            fs.clear();

            // Launching 4 (i.e. same as number of threads) tasks may deadlock
            // as no thread is available to steal from the current thread.
            fs.push_back(tp.suspend_processing_unit(thread_num));
            fs.push_back(tp.suspend_processing_unit(thread_num));
            fs.push_back(tp.suspend_processing_unit(thread_num));

            hpx::wait_all(fs);

            fs.clear();

            // Launching 4 (i.e. same as number of threads) tasks may deadlock
            // as no thread is available to steal from the current thread.
            fs.push_back(tp.resume_processing_unit(thread_num));
            fs.push_back(tp.resume_processing_unit(thread_num));
            fs.push_back(tp.resume_processing_unit(thread_num));

            hpx::wait_all(fs);
        }
    }

    {
        // Check random scheduling with reducing resources.
        std::size_t thread_num = 0;
        bool up = true;
        std::vector<hpx::future<void>> fs;
        hpx::util::high_resolution_timer t;
        while (t.elapsed() < 2)
        {
            for (std::size_t i = 0;
                i < hpx::resource::get_num_threads("default") * 10;
                ++i)
            {
                fs.push_back(hpx::async([](){}));
            }

            if (up)
            {
                if (thread_num != hpx::resource::get_num_threads("default") - 1)
                {
                    tp.suspend_processing_unit(thread_num).get();
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
                tp.resume_processing_unit(thread_num - 1).get();

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
            tp.resume_processing_unit(thread_num_resume).get();
        }
    }

    return hpx::finalize();
}

template <typename Scheduler>
void test_scheduler(int argc, char* argv[])
{
    std::vector<std::string> cfg =
    {
        "hpx.os_threads=4"
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
                hpx::threads::policies::delay_exit |
                hpx::threads::policies::enable_elasticity |
                hpx::threads::policies::enable_suspension);

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
    // NOTE: Static schedulers do not support suspending the own worker thread
    // because they do not steal work.

    test_scheduler<hpx::threads::policies::local_queue_scheduler<>>(argc, argv);
    test_scheduler<hpx::threads::policies::local_priority_queue_scheduler<>>(argc,
        argv);

    return hpx::util::report_errors();
}

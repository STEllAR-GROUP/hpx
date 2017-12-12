//  Copyright (c) 2017 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

int hpx_main(int argc, char* argv[])
{
    std::size_t const num_threads = hpx::resource::get_num_threads("default");

    HPX_TEST_EQ(std::size_t(4), num_threads);

    hpx::threads::detail::thread_pool_base& tp =
        hpx::resource::get_thread_pool("default");

    HPX_TEST_EQ(tp.get_active_os_thread_count(), std::size_t(4));

    {
        // Try suspending without elasticity enabled, should throw an exception
        try
        {
            tp.suspend_processing_unit(0);
            HPX_TEST_MSG(false, "Suspending should not be allowed with "
                 "elasticity disabled");
        }
        catch (std::runtime_error const&)
        {
        }
    }

    // Enable elasticity
    tp.set_scheduler_mode(
        hpx::threads::policies::scheduler_mode(
            hpx::threads::policies::do_background_work |
            hpx::threads::policies::reduce_thread_priority |
            hpx::threads::policies::delay_exit |
            hpx::threads::policies::enable_elasticity));

    {
        // Check number of used resources
        for (std::size_t thread_num = 0; thread_num < num_threads - 1; ++thread_num)
        {
            tp.suspend_processing_unit(thread_num);
            HPX_TEST_EQ(std::size_t(num_threads - thread_num - 1),
                tp.get_active_os_thread_count());
        }

        for (std::size_t thread_num = 0; thread_num < num_threads - 1; ++thread_num)
        {
            tp.resume_processing_unit(thread_num);
            HPX_TEST_EQ(std::size_t(thread_num + 2),
                tp.get_active_os_thread_count());
        }
    }

    {
        // Check suspending pu on which current thread is running
        std::size_t worker_thread_num = hpx::get_worker_thread_num();
        tp.suspend_processing_unit(worker_thread_num);
        tp.resume_processing_unit(worker_thread_num);
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
                    tp.suspend_processing_unit(thread_num_suspend);
                }
            }

            hpx::async(test_function).get();

            for (std::size_t thread_num_resume = 0;
                thread_num_resume < num_threads;
                ++thread_num_resume)
            {
                if (thread_num != thread_num_resume)
                {
                    tp.resume_processing_unit(thread_num_resume);
                }
            }
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
                    tp.suspend_processing_unit(thread_num);
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
                tp.resume_processing_unit(thread_num - 1);

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
            tp.resume_processing_unit(thread_num_resume);
        }
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg =
    {
        "hpx.os_threads=4"
    };

    using hpx::resource::scheduling_policy;

    std::vector<scheduling_policy> const policies =
    {
        scheduling_policy::local,
        scheduling_policy::local_priority_fifo,
        scheduling_policy::local_priority_lifo,
        // NOTE: Static scheduling policies do not support suspending the own
        // worker thread because they do not steal work.
        //scheduling_policy::static_,
        //scheduling_policy::static_priority,
        scheduling_policy::abp_priority,
        scheduling_policy::hierarchy,
        //scheduling_policy::periodic_priority,
    };

    for (auto policy : policies)
    {
        // Set up the resource partitioner
        hpx::resource::partitioner rp(argc, argv, std::move(cfg));
        rp.create_thread_pool("default", policy);

        HPX_TEST_EQ(hpx::init(argc, argv), 0);
    }

    return hpx::util::report_errors();
}

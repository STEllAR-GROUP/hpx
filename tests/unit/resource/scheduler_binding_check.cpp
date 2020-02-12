//  Copyright (c) 2020 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Test that loops over each core assigned to the program and launches
// tasks bound to that core incrementally.
// Tasks should always report the right core number when they run.

#include <hpx/async.hpp>
#include <hpx/debugging/print.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/runtime/threads/detail/scheduled_thread_pool_impl.hpp>
#include <hpx/runtime/threads/executors/default_executor.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>
#include <hpx/runtime/threads/policies/shared_priority_queue_scheduler.hpp>
#include <hpx/testing.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

namespace hpx {
    // use <true>/<false> to enable/disable debug printing
    using sbc_print_on = hpx::debug::enable_print<false>;
    static sbc_print_on deb_schbin("SCHBIND");
}    // namespace hpx

// counts down on destruction
struct dec_counter
{
    explicit dec_counter(std::atomic<int>& counter)
      : counter_(counter)
    {
    }
    ~dec_counter()
    {
        --counter_;
    }
    //
    std::atomic<int>& counter_;
};

void threadLoop()
{
    const unsigned iterations = 2048;
    std::atomic<int> count_down(iterations);

    auto f = [&count_down](std::size_t thread_expected) {
        dec_counter dec(count_down);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::size_t thread_actual = hpx::get_worker_thread_num();
        hpx::deb_schbin.debug(hpx::debug::str<>("Running on thread"),
            thread_actual, "Expected", thread_expected);
        if (thread_actual != thread_expected)
        {
            hpx::deb_schbin.error(hpx::debug::str<>("actual!=expected"), "Got",
                thread_actual, "Expected", thread_expected);
        }
        HPX_TEST_EQ(thread_actual, thread_expected);
    };

    std::size_t threads = hpx::get_num_worker_threads();
    // launch tasks on threads using numbering 0,1,2,3...0,1,2,3
    for (std::size_t i = 0; i < iterations; ++i)
    {
        auto exec = hpx::threads::executors::default_executor(
            hpx::threads::thread_priority_bound,
            hpx::threads::thread_stacksize_default,
            hpx::threads::thread_schedule_hint(std::int16_t(i % threads)));
        hpx::async(exec, f, (i % threads)).get();
    }

    do
    {
        hpx::this_thread::yield();
    } while (count_down > 0);

    return;
}

int hpx_main(boost::program_options::variables_map&)
{
    auto const current = hpx::threads::get_self_id_data()->get_scheduler_base();
    std::cout << "Scheduler is " << current->get_description() << std::endl;
    if (std::string("core-shared_priority_queue_scheduler") !=
        current->get_description())
    {
        std::cout << "The scheduler might not work properly " << std::endl;
    }

    threadLoop();
    hpx::finalize();
    return hpx::util::report_errors();
}

int main(int argc, char* argv[])
{
    // Configure application-specific options.
    hpx::program_options::options_description desc_cmdline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // Create the resource partitioner
    hpx::resource::partitioner rp(desc_cmdline, argc, argv);

    // setup the default pool with a numa/binding aware scheduler
    rp.create_thread_pool("default",
        hpx::resource::scheduling_policy::shared_priority,
        hpx::threads::policies::scheduler_mode(
            hpx::threads::policies::default_mode));

    return hpx::init(desc_cmdline, argc, argv);
}

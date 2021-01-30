//  Copyright (c) 2020 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Test that loops over each core assigned to the program and launches
// tasks bound to that core incrementally.
// Tasks should always report the right core number when they run.

#include <hpx/debugging/print.hpp>
#include <hpx/execution.hpp>
#include <hpx/future.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/schedulers.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/runtime.hpp>
#include <hpx/thread.hpp>

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
    unsigned const iterations = 2048;
    std::atomic<int> count_down(iterations);

    auto f = [&count_down](std::size_t iteration, std::size_t thread_expected) {
        dec_counter dec(count_down);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::size_t thread_actual = hpx::get_worker_thread_num();
        hpx::deb_schbin.debug(hpx::debug::str<10>("Iteration"),
            hpx::debug::dec<4>(iteration),
            hpx::debug::str<20>("Running on thread"), thread_actual,
            hpx::debug::str<10>("Expected"), thread_expected);
        HPX_TEST_EQ(thread_actual, thread_expected);
    };

    std::size_t threads = hpx::get_num_worker_threads();
    // launch tasks on threads using numbering 0,1,2,3...0,1,2,3
    for (std::size_t i = 0; i < iterations; ++i)
    {
        auto exec = hpx::execution::parallel_executor(
            hpx::threads::thread_priority::bound,
            hpx::threads::thread_stacksize::default_,
            hpx::threads::thread_schedule_hint(std::int16_t(i % threads)));
        hpx::async(exec, f, i, (i % threads)).get();
    }

    do
    {
        hpx::this_thread::yield();
        hpx::deb_schbin.debug(
            hpx::debug::str<15>("count_down"), hpx::debug::dec<4>(count_down));
    } while (count_down > 0);

    hpx::deb_schbin.debug(
        hpx::debug::str<15>("complete"), hpx::debug::dec<4>(count_down));
    HPX_TEST_EQ(count_down, 0);
}

int hpx_main()
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
    hpx::deb_schbin.debug(hpx::debug::str<15>("Finalized"));
    return hpx::util::report_errors();
}

int main(int argc, char* argv[])
{
    hpx::init_params init_args;

    init_args.rp_callback = [](auto& rp,
                                hpx::program_options::variables_map const&) {
        // setup the default pool with a numa/binding aware scheduler
        rp.create_thread_pool("default",
            hpx::resource::scheduling_policy::shared_priority,
            hpx::threads::policies::scheduler_mode(
                hpx::threads::policies::default_mode));
    };

    return hpx::init(argc, argv, init_args);
}

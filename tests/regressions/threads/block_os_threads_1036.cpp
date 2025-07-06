//  Copyright (c) 2011-2013 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test demonstrates the issue described in #1036: Scheduler hangs when
// user code attempts to "block" OS-threads

#include <hpx/barrier.hpp>
#include <hpx/functional.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/timing.hpp>
#include <hpx/modules/topology.hpp>
#include <hpx/thread.hpp>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
void blocker(std::shared_ptr<hpx::barrier<>> exit_barrier,
    std::atomic<std::uint64_t>& entered, std::atomic<std::uint64_t>& started,
    std::unique_ptr<std::atomic<std::uint64_t>[]>& blocked_threads,
    std::uint64_t worker)
{
    // reschedule if we are not on the correct OS thread...
    if (worker != hpx::get_worker_thread_num())
    {
        hpx::threads::thread_init_data data(
            hpx::threads::make_thread_function_nullary(
                hpx::bind(&blocker, exit_barrier, std::ref(entered),
                    std::ref(started), std::ref(blocked_threads), worker)),
            "blocker", hpx::threads::thread_priority::normal,
            hpx::threads::thread_schedule_hint(
                static_cast<std::int16_t>(worker)));
        hpx::threads::register_work(data);
        return;
    }

    blocked_threads[hpx::get_worker_thread_num()].fetch_add(1);

    entered.fetch_add(1);

    HPX_TEST_EQ(worker, hpx::get_worker_thread_num());

    while (started.load() != 1)
        continue;

    exit_barrier->arrive_and_drop();
}

///////////////////////////////////////////////////////////////////////////////
std::uint64_t delay = 100;

int hpx_main()
{
    {
        ///////////////////////////////////////////////////////////////////////
        // Block all other OS threads.
        std::atomic<std::uint64_t> entered(0);
        std::atomic<std::uint64_t> started(0);

        std::uint64_t const os_thread_count = hpx::get_os_thread_count();

        std::shared_ptr<hpx::barrier<>> exit_barrier =
            std::make_shared<hpx::barrier<>>(os_thread_count);

        std::unique_ptr<std::atomic<std::uint64_t>[]> blocked_threads(
            new std::atomic<std::uint64_t>[os_thread_count]);

        for (std::uint64_t i = 0; i < os_thread_count; ++i)
            blocked_threads[i].store(0);

        std::uint64_t scheduled = 0;
        for (std::uint64_t i = 0; i < os_thread_count; ++i)
        {
            if (i == hpx::get_worker_thread_num())
                continue;

            hpx::threads::thread_init_data data(
                hpx::threads::make_thread_function_nullary(
                    hpx::bind(&blocker, exit_barrier, std::ref(entered),
                        std::ref(started), std::ref(blocked_threads), i)),
                "blocker", hpx::threads::thread_priority::normal,
                hpx::threads::thread_schedule_hint(
                    static_cast<std::int16_t>(i)));
            hpx::threads::register_work(data);
            ++scheduled;
        }
        HPX_TEST_EQ(scheduled, os_thread_count - 1);

        while (entered.load() != (os_thread_count - 1))
            continue;

        {
            double delay_sec = static_cast<double>(delay) * 1e-6;
            hpx::chrono::high_resolution_timer td;

            while (true)
            {
                if (td.elapsed() > delay_sec)
                    break;
            }
        }

        started.fetch_add(1);

        for (std::uint64_t i = 0; i < os_thread_count; ++i)
            HPX_TEST_LTE(blocked_threads[i].load(), std::uint64_t(1));

        exit_barrier->arrive_and_wait();
    }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    using namespace hpx::program_options;

    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()("delay",
        value<std::uint64_t>(&delay)->default_value(100),
        "time in micro-seconds for the delay loop");

    // We force this test to use all available threads by default.
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX.
    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}

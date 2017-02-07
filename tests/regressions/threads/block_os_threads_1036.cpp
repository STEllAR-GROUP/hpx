//  Copyright (c) 2011-2013 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test demonstrates the issue described in #1036: Scheduler hangs when
// user code attempts to "block" OS-threads

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/topology.hpp>

#include <boost/atomic.hpp>
#include <boost/scoped_array.hpp>

#include <cstdint>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
void blocker(
    boost::atomic<std::uint64_t>* entered
  , boost::atomic<std::uint64_t>* started
  , boost::scoped_array<boost::atomic<std::uint64_t> >* blocked_threads
  , std::size_t worker
    )
{
    // reschedule if we are not on the correct OS thread...
    if (worker != hpx::get_worker_thread_num())
    {
        hpx::threads::register_work(
            hpx::util::bind(&blocker, entered, started, blocked_threads, worker),
            "blocker", hpx::threads::pending,
            hpx::threads::thread_priority_normal, worker);
        return;
    }

    (*blocked_threads)[hpx::get_worker_thread_num()].fetch_add(1);

    entered->fetch_add(1);

    HPX_TEST_EQ(worker, hpx::get_worker_thread_num());

    while (started->load() != 1)
        continue;
}

///////////////////////////////////////////////////////////////////////////////
volatile int i = 0;
std::uint64_t delay = 100;

int hpx_main()
{
    {
        ///////////////////////////////////////////////////////////////////////
        // Block all other OS threads.
        boost::atomic<std::uint64_t> entered(0);
        boost::atomic<std::uint64_t> started(0);

        std::uint64_t const os_thread_count = hpx::get_os_thread_count();

        boost::scoped_array<boost::atomic<std::uint64_t> >
            blocked_threads(
                new boost::atomic<std::uint64_t>[os_thread_count]);

        for (std::uint64_t i = 0; i < os_thread_count; ++i)
            blocked_threads[i].store(0);

        std::size_t scheduled = 0;
        for (std::uint64_t i = 0; i < os_thread_count; ++i)
        {
            if (i == hpx::get_worker_thread_num())
                continue;

            hpx::threads::register_work(
                hpx::util::bind(&blocker, &entered, &started, &blocked_threads, i),
                "blocker", hpx::threads::pending,
                hpx::threads::thread_priority_normal, i);
            ++scheduled;
        }
        HPX_TEST_EQ(scheduled, os_thread_count - 1);


        while (entered.load() != (os_thread_count - 1))
            continue;

        {
            double delay_sec = delay * 1e-6;
            hpx::util::high_resolution_timer td;

            while (true)
            {
                if (td.elapsed() > delay_sec)
                    break;
                else
                    ++i;
            }
        }

        started.fetch_add(1);

        for (std::uint64_t i = 0; i < os_thread_count; ++i)
            HPX_TEST(blocked_threads[i].load() <= 1);
    }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(
    int argc
  , char* argv[]
    )
{
    using namespace boost::program_options;

    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ( "delay"
        , value<std::uint64_t>(&delay)->default_value(100)
        , "time in micro-seconds for the delay loop")
        ;

    // We force this test to use all available threads by default.
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX.
    return hpx::init(cmdline, argc, argv, cfg);
}



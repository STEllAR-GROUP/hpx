//  Copyright (c) 2011-2013 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test demonstrates the issue described in #1036: Scheduler hangs when
// user code attempts to "block" OS-threads 

#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/topology.hpp>

#include <boost/assign/std/vector.hpp>

///////////////////////////////////////////////////////////////////////////////
void blocker(
    boost::atomic<bool>& entered
  , boost::atomic<bool>& started
    )
{
    entered = true;

    while (!started)
        continue;
}

///////////////////////////////////////////////////////////////////////////////
volatile int i = 0;
boost::uint64_t delay = 100;

int hpx_main()
{
    {
        ///////////////////////////////////////////////////////////////////////
        // Block all other OS threads.
        boost::atomic<bool> started(false);

        boost::uint64_t num_threads = hpx::get_os_thread_count() - 1;
        for (boost::uint64_t j = 0; j != num_threads; ++j)
        {
            boost::atomic<bool> entered(false);

            hpx::threads::register_work(
                boost::bind(&blocker, boost::ref(entered), boost::ref(started)),
                "blocker", hpx::threads::pending, 
                hpx::threads::thread_priority_normal);

            while (!entered)
                continue;
        }

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

        started = true;
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
        , value<boost::uint64_t>(&delay)->default_value(100)
        , "time in micro-seconds for the delay loop")
        ;

    // We force this test to use all available threads by default.
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency());

    // Initialize and run HPX.
    return hpx::init(cmdline, argc, argv, cfg);
}



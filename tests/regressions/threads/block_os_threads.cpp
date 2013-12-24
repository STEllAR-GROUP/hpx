//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>

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

int hpx_main(
    boost::program_options::variables_map& vm
    )
{
    {
        ///////////////////////////////////////////////////////////////////////
        // Block all other OS threads.
        boost::atomic<bool> started(false);

        for (boost::uint64_t i = 0; i < (hpx::get_os_thread_count() - 1); ++i)
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

    // Initialize and run HPX.
    return hpx::init(cmdline, argc, argv);
}



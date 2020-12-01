//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/modules/timing.hpp>

#include <chrono>
#include <iostream>

using hpx::program_options::variables_map;
using hpx::program_options::options_description;
using hpx::program_options::value;

using std::chrono::seconds;

using hpx::threads::get_self_id;
using hpx::threads::get_self;
using hpx::threads::set_thread_state;

using hpx::chrono::high_resolution_timer;

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    {
        std::cout << "waiting for 5 seconds\n";

        high_resolution_timer t;

        // Schedule a wakeup in 5 seconds.
        set_thread_state(get_self_id(), seconds(5),
            hpx::threads::thread_schedule_state::pending);

        // Suspend this HPX thread.
        hpx::this_thread::suspend(
            hpx::threads::thread_schedule_state::suspended);

        std::cout << "woke up after " << t.elapsed() << " seconds\n";
    }

    hpx::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX.
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return hpx::init(argc, argv, init_args);
}


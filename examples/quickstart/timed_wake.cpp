//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <chrono>
#include <iostream>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using std::chrono::seconds;

using hpx::init;
using hpx::finalize;

using hpx::threads::pending;
using hpx::threads::suspended;
using hpx::threads::get_self_id;
using hpx::threads::get_self;
using hpx::threads::set_thread_state;

using hpx::util::high_resolution_timer;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        std::cout << "waiting for 5 seconds\n";

        high_resolution_timer t;

        // Schedule a wakeup in 5 seconds.
        set_thread_state(get_self_id(), seconds(5), pending);

        // Suspend this HPX thread.
        hpx::this_thread::suspend(suspended);

        std::cout << "woke up after " << t.elapsed() << " seconds\n";
    }

    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX.
    return init(desc_commandline, argc, argv);
}


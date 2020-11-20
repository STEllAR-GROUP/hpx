//  Copyright (c) 2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example demonstrates the use of the utility function
// make_ready_future_after to orchestrate timed operations with 'normal'
// asynchronous work.

#include <hpx/hpx_init.hpp>
#include <hpx/iostream.hpp>
#include <hpx/modules/timing.hpp>

#include <chrono>

///////////////////////////////////////////////////////////////////////////////
void wake_up_after_2_seconds()
{
    hpx::cout << "waiting for 2 seconds\n";

    hpx::chrono::high_resolution_timer t;

    // Schedule a wakeup after 2 seconds.
    using std::chrono::seconds;
    hpx::future<void> f = hpx::make_ready_future_after(seconds(2));

    // ... do other things while waiting for the future to get ready

    // wait until the new future gets ready
    f.wait();

    hpx::cout << "woke up after " << t.elapsed()
              << " seconds\n" << hpx::flush;
}

int return_int_at_time()
{
    hpx::cout << "generating an 'int' value 2 seconds from now\n";

    hpx::chrono::high_resolution_timer t;

    // Schedule a wakeup 2 seconds from now.
    using namespace std::chrono;
    hpx::future<int> f = hpx::make_ready_future_at(
        steady_clock::now() + seconds(2), 42);

    // ... do other things while waiting for the future to get ready

    // wait until the new future gets ready (should return 42)
    int retval = f.get();

    hpx::cout << "woke up after " << t.elapsed()
              << " seconds, returned: " << retval << "\n"
              << hpx::flush;

    return retval;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    wake_up_after_2_seconds();
    return_int_at_time();
    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Initialize and run HPX.
    return hpx::init(argc, argv);
}

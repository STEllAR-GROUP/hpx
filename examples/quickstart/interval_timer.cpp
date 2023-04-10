//  Copyright (c) 2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example demonstrates the use of the utility function
// make_ready_future_after to orchestrate timed operations with 'normal'
// asynchronous work.

#include <hpx/init.hpp>
#include <hpx/modules/runtime_local.hpp>
#include <hpx/thread.hpp>

#include <chrono>
#include <iostream>

///////////////////////////////////////////////////////////////////////////////
bool call_every_500_millisecs()
{
    static int counter = 0;

    std::cout << "Callback " << ++counter << std::endl;
    return counter != 10;    // stop timer after 10 invocations
}

int hpx_main()
{
    {
        // initialize timer to invoke given function every 500 milliseconds
        hpx::util::interval_timer timer(
            &call_every_500_millisecs, std::chrono::milliseconds(500));

        timer.start();

        // wait for timer to have invoked the function 10 times
        while (!timer.is_terminated())
            hpx::this_thread::yield();
    }

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::local::init(hpx_main, argc, argv);
}

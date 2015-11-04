//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example demonstrates the use of the utility function
// make_ready_future_after to orchestrate timed operations with 'normal'
// asynchronous work.

#include <hpx/hpx_main.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/iostreams.hpp>

#include <boost/chrono/chrono.hpp>

///////////////////////////////////////////////////////////////////////////////
bool call_every_500_millisecs()
{
    static int counter = 0;

    hpx::cout << "Callback " << ++counter << std::endl;
    return counter != 10;     // stop timer after 10 invocations
}

int main()
{
    {
        // initialize timer to invoke given function every 500 milliseconds
        hpx::util::interval_timer timer(
            &call_every_500_millisecs, boost::chrono::milliseconds(500)
        );

        timer.start();

        // wait for timer to have invoked the function 10 times
        while (!timer.is_terminated())
            hpx::this_thread::yield();
    }

    return hpx::finalize();
}


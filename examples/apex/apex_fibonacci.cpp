////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// Naive SMP version implemented with futures.

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/util.hpp>
#include <apex_api.hpp>

#include <iostream>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>

///////////////////////////////////////////////////////////////////////////////
//[fib_action
// forward declaration of the Fibonacci function
boost::uint64_t fibonacci(boost::uint64_t n);

// This is to generate the required boilerplate we need for the remote
// invocation to work.
HPX_PLAIN_ACTION(fibonacci, fibonacci_action);
//]

///////////////////////////////////////////////////////////////////////////////
//[fib_func
boost::uint64_t fibonacci(boost::uint64_t n)
{
    if (n < 2)
        return n;

    // We restrict ourselves to execute the Fibonacci function locally.
    hpx::naming::id_type const locality_id = hpx::find_here();

    // Invoking the Fibonacci algorithm twice is inefficient.
    // However, we intentionally demonstrate it this way to create some
    // heavy workload.

    fibonacci_action fib;
    hpx::future<boost::uint64_t> n1 =
        hpx::async(fib, locality_id, n - 1);
    hpx::future<boost::uint64_t> n2 =
        hpx::async(fib, locality_id, n - 2);

    return n1.get() + n2.get();   // wait for the Futures to return their values
}
//]

///////////////////////////////////////////////////////////////////////////////
//[fib_hpx_main
int hpx_main(boost::program_options::variables_map& vm)
{
    // extract command line argument, i.e. fib(N)
    boost::uint64_t n = vm["n-value"].as<boost::uint64_t>();

    {
        // Keep track of the time required to execute.
        hpx::util::high_resolution_timer t;

        // Wait for fib() to return the value
        fibonacci_action fib;
        boost::uint64_t r = fib(hpx::find_here(), n);

        char const* fmt = "fibonacci(%1%) == %2%\nelapsed time: %3% [s]\n";
        std::cout << (boost::format(fmt) % n % r % t.elapsed());
    }

    return hpx::finalize(); // Handles HPX shutdown
}
//]

///////////////////////////////////////////////////////////////////////////////
//[fib_main
int main(int argc, char* argv[])
{
    apex::apex_options::use_screen_output(true);

    // Configure application-specific options
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "n-value",
          boost::program_options::value<boost::uint64_t>()->default_value(10),
          "n value for the Fibonacci function")
        ;

    std::set<apex_event_type> when = {APEX_STARTUP, APEX_SHUTDOWN, APEX_NEW_NODE,
        APEX_NEW_THREAD, APEX_START_EVENT, APEX_STOP_EVENT, APEX_SAMPLE_VALUE};
    apex::register_policy(when, [](apex_context const& context)->int{
        switch(context.event_type) {
            case APEX_STARTUP: {
              std::cout << "Startup event" << std::endl;
              break;
            }
            case APEX_SHUTDOWN: {
              std::cout << "Shutdown event" << std::endl;
              break;
            }
            case APEX_NEW_NODE: {
              std::cout << "New node event" << std::endl;
              break;
            }
            case APEX_NEW_THREAD: {
              std::cout << "New thread event" << std::endl;
              break;
            }
            case APEX_START_EVENT: {
              std::cout << "Start event" << std::endl;
              break;
            }
            case APEX_STOP_EVENT: {
              std::cout << "Stop event" << std::endl;
              break;
            }
            case APEX_SAMPLE_VALUE: {
              std::cout << "Sample value event" << std::endl;
              break;
            }
            default: {
              std::cout << "Unknown event" << std::endl;
            }
        }
        return APEX_NOERROR;
    });

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}
//]

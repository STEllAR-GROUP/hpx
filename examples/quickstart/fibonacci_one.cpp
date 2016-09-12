////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// Naive SMP version implemented with futures (but still a bit more
// sophisticated than fibonacci.cpp).

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/util.hpp>

#include <cstdint>
#include <iostream>

#include <boost/format.hpp>

///////////////////////////////////////////////////////////////////////////////
//[fib_action
// forward declaration of the Fibonacci function
std::uint64_t fibonacci(std::uint64_t n);

// This is to generate the required boilerplate we need for the remote
// invocation to work.
HPX_PLAIN_ACTION(fibonacci, fibonacci_action);
//]

///////////////////////////////////////////////////////////////////////////////
//[fib_func
std::uint64_t fibonacci(std::uint64_t n)
{
    if (n < 2)
        return n;

    // We restrict ourselves to execute the Fibonacci function locally.
    hpx::naming::id_type const locality_id = hpx::find_here();

    // Run one branch of the Fibonacci calculation on this thread, while the
    // other branch is scheduled in a separate thread.
    hpx::future<std::uint64_t> n1 =
        hpx::async<fibonacci_action>(locality_id, n-1);
    std::uint64_t n2 = fibonacci(n-2);

    return n1.get() + n2;   // wait for the Future to return its values
}
//]

std::uint64_t fibonacci_direct(std::uint64_t n)
{
    if (n < 2)
        return n;

    // Run one branch of the Fibonacci calculation on this thread, while the
    // other branch is scheduled in a separate thread.
    hpx::future<std::uint64_t> n1 =
        hpx::async(&fibonacci_direct, n-1);
    std::uint64_t n2 = fibonacci(n-2);

    return n1.get() + n2;   // wait for the Future to return its values
}

///////////////////////////////////////////////////////////////////////////////
//[fib_hpx_main
int hpx_main(boost::program_options::variables_map& vm)
{
    // extract command line argument, i.e. fib(N)
    std::uint64_t n = vm["n-value"].as<std::uint64_t>();

    {
        // Keep track of the time required to execute.
        hpx::util::high_resolution_timer t;

        // Create a Future for the whole calculation, execute it locally, and
        // wait for it.
        hpx::future<std::uint64_t> f =
            hpx::async<fibonacci_action>(hpx::find_here(), n);

        // wait for future f to return value
        std::uint64_t r = f.get();

        char const* fmt = "fibonacci(%1%) == %2%, elapsed time: %3% [s]\n";
        std::cout << (boost::format(fmt) % n % r % t.elapsed());
    }

    {
        // Keep track of the time required to execute.
        hpx::util::high_resolution_timer t;

        std::uint64_t r = fibonacci_direct(n);;

        char const* fmt = "fibonacci_direct(%1%) == %2%, elapsed time: %3% [s]\n";
        std::cout << (boost::format(fmt) % n % r % t.elapsed());
    }

    return hpx::finalize(); // Handles HPX shutdown
}
//]

///////////////////////////////////////////////////////////////////////////////
//[fib_main
int main(int argc, char* argv[])
{
    // Configure application-specific options
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "n-value",
          boost::program_options::value<std::uint64_t>()->default_value(10),
          "n value for the Fibonacci function")
        ;

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}
//]
